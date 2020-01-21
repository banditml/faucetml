import math
import threading
import time
from typing import Dict, Generator, List, NoReturn, Tuple
import queue

from google.cloud import bigquery
from google.oauth2 import service_account
from google.cloud.bigquery.job import QueryJob
import pandas as pd
from retry import retry

from .utils import get_logger


logger = get_logger(__name__)


class DataStore:
    bigquery = "bigquery"


def get_data_reader(
    datastore: str,
    credential_path: str,
    hash_on_feature: str,
    table_name: str,
    ds: str = None,
    epochs: int = 1,
    batch_size: int = 1024,
    chunk_size: int = 1024 * 100,
    exclude_features: List[str] = None,
    table_sample_percent: float = None,
    test_split_percent: float = None,
    skip_small_batches: bool = True,
):
    """Get a DataReader object.

    Args:
        datastore: Datastore fetching data from (i.e. "bigquery").
        credential_path: Path to credentials for datastore.
        hash_on_feature: Feature to hash on for random sampling. Used to split
            dataset into training and test.
        table_name: Name of table to fetch data from.
        ds: Date for table partition.
        epochs: Number of epochs to train for.
        batch_size: Mini-batch size in training.
        chunk_size: Size of data to fetch from table. Should be >> batch_size
            but small enough to fit in memory.
        exclude_features: Features to exclude in training.
        table_sample_percent: Percent of table to use for training and eval.
        test_split_percent: Percent of table sample to use for eval.

    Returns:
        DataReader object.

    """

    if datastore == DataStore.bigquery:
        return BigQueryReader(
            datastore,
            credential_path,
            hash_on_feature,
            table_name,
            ds,
            epochs,
            batch_size,
            chunk_size,
            exclude_features,
            table_sample_percent,
            test_split_percent,
            skip_small_batches,
        )
    else:
        raise Exception("Datastore {} not supported".format(kwargs["datastore"]))


class DataReader:
    def __init__(
        self,
        datastore: str,
        credential_path: str,
        hash_on_feature: str,
        table_name: str,
        ds: str,
        epochs: int,
        batch_size: int,
        chunk_size: int,
        exclude_features: List[str],
        table_sample_percent: float,
        test_split_percent: float,
        skip_small_batches: bool,
    ):
        # Intialize client & set table properties
        self.client = self._get_client(credential_path)
        self.table_name = table_name
        self.exclude_features = exclude_features

        # Figure out sampling + train & test
        self.max_row_filter = table_sample_percent / 100
        self.train_set_filter = self.max_row_filter * (1 - test_split_percent / 100)

        # Generate queries for the training & test set
        self.train_query, self.eval_query = self._build_train_and_eval_queries(
            hash_on_feature, ds
        )

        # Create query jobs to create temp tables (non-blocking)
        self.train_query_job = self._create_tmp_table(train=True)
        self.eval_query_job = self._create_tmp_table(train=False)

        # Minibatch settings
        self.batch_size = batch_size
        self.skip_small_batches = skip_small_batches

        # Chunk settings
        self.chunk_size = chunk_size
        assert (
            chunk_size % batch_size == 0
        ), f"Chunk size: {chunk_size} must be multiple of batch size: {batch_size}."

        # Epoch settings
        self.epochs = epochs
        self.current_epoch_num = 0

    def _get_client(self, credential_path: str):
        raise NotImplementedError

    def _build_train_and_eval_queries(
        self, hash_on_feature: str, ds: str = None
    ) -> str:
        raise NotImplementedError

    def _create_tmp_table(self, train: bool = True):
        raise NotImplementedError

    def _create_query_iterator(self, train: bool = True):
        raise NotImplementedError

    def _batches_per_epoch(self, num_rows: int) -> int:
        """Computes number of batches per epoch."""
        num_full_chunks = math.floor(num_rows / self.chunk_size)
        full_chunk_batches = num_full_chunks * (self.chunk_size / self.batch_size)
        left_over_rows = num_rows % self.chunk_size
        if self.skip_small_batches:
            return int(
                full_chunk_batches + math.floor(left_over_rows / self.batch_size)
            )
        return int(full_chunk_batches + math.ceil(left_over_rows / self.batch_size))

    def _execute_chunk_job_and_add_to_queue(self) -> NoReturn:
        """Fetch a chunk and add it the queue. This is blocking."""
        try:
            # Generators must be locked if accessing them in a multi-threaded way.
            self.lock.acquire()
            chunk = next(self.chunk_gen)
            self.lock.release()
            self.chunk_q.put(chunk)
        except StopIteration:
            # no more chunks left to iterate through
            pass
        except Exception as e:
            logger.error(f"Failed getting chunk due to exception {e}.")

    def _background_put_chunk_in_queue(self) -> NoReturn:
        """Create a thread and use it to put a chunk in the queue."""
        thread = threading.Thread(target=self._execute_chunk_job_and_add_to_queue)
        thread.daemon = True
        thread.start()

    def _get_batch_from_chunk(
        self, chunk_df: pd.DataFrame, start_idx: int, end_idx: int, eval=False
    ) -> Tuple[Dict, bool]:
        "Fetch a minibatch of data from a cached chunk of data."

        end_idx = min(len(chunk_df), end_idx)
        if self.skip_small_batches and (end_idx - start_idx) < self.batch_size:
            return None, True

        return_dict = {
            "features_df": pd.DataFrame(chunk_df.features.values.tolist())[
                start_idx:end_idx
            ],
            "labels_df": pd.DataFrame(chunk_df.labels.values.tolist())[
                start_idx:end_idx
            ],
            "batch_num": (self.current_batch_num + 1)
            + ((self.current_epoch_num - 1) * self.batches_per_epoch),
            "batches_per_epoch": self.batches_per_epoch,
        }

        if eval:
            percent_done = round(
                (self.current_batch_num + 1) / (self.batches_per_epoch) * 100
            )
        else:
            percent_done = round(
                (
                    (self.current_batch_num + 1)
                    + ((self.current_epoch_num - 1) * self.batches_per_epoch)
                )
                / (self.batches_per_epoch * self.epochs)
                * 100
            )

        logger.info(
            f"Got batch {self.current_batch_num + 1}/{self.batches_per_epoch} "
            f"for epoch {self.current_epoch_num}/{self.epochs} ({percent_done}%)"
        )

        self.batch_start_idx = end_idx
        self.batch_end_idx = self.batch_start_idx + self.batch_size
        self.current_batch_num += 1
        chunk_done = self.batch_start_idx >= len(chunk_df)
        return return_dict, chunk_done

    def _set_epoch_state(self, train: bool = True, warm_up_chunks: int = 2) -> int:
        num_rows, self.chunk_gen = self._create_query_iterator(train=train)
        self.lock = threading.Lock()
        self.current_chunk = None
        self.num_chunks = math.ceil(num_rows / self.chunk_size)
        self.completed_chunks = 0
        self.failed_chunks = 0
        self.chunk_q = queue.Queue()
        self.chunk_job_q = queue.Queue()
        self.current_batch_num = 0
        self.batches_per_epoch = self._batches_per_epoch(num_rows)
        self.batch_start_idx = 0
        self.batch_end_idx = self.batch_start_idx + self.batch_size

        for i in range(warm_up_chunks):
            self._background_put_chunk_in_queue()

        return num_rows

    def prep_for_eval(self) -> NoReturn:
        """Set up for an eval epoch. Must be called before start of eval pass."""
        num_rows = self._set_epoch_state(train=False)
        logger.info("*" * 50)
        logger.info("Starting end of epoch evaluation...")
        logger.info("*" * 50)
        logger.info(f"Eval pass {self.current_epoch_num} contains {num_rows} rows.")

    def prep_for_epoch(self) -> NoReturn:
        """Set up for a training epoch. Must be called before start of a training epoch."""
        self.current_epoch_num += 1
        num_rows = self._set_epoch_state(train=True)
        logger.info(f"Epoch {self.current_epoch_num} contains {num_rows} rows.")

    def get_batch(self, eval: bool = False) -> Dict:
        """Get a batch of data from the table.

        Args:
            eval: Whether or not this batch is used for evaluation.

        Returns:
            Dict {
                "features_df": pd.DataFrame,  # dataframe of featuers
                "labels_df": pd.DataFrame,  # dataframe of labels
                "batch_num": int,  # current batch number
                "batches_per_epoch": int,  # batches in the epoch
            }

        """
        if (self.completed_chunks + self.failed_chunks) == self.num_chunks:
            return None

        if self.current_chunk is None:
            self.current_chunk = self.chunk_q.get()
            # replace what you just took
            self._background_put_chunk_in_queue()

        batch_dict, chunk_done = self._get_batch_from_chunk(
            self.current_chunk, self.batch_start_idx, self.batch_end_idx, eval
        )

        if chunk_done:
            self.current_chunk = None
            self.batch_start_idx = 0
            self.batch_end_idx = self.batch_start_idx + self.batch_size
            self.completed_chunks += 1

        return batch_dict


class BigQueryReader(DataReader):
    def __init__(
        self,
        datastore: str,
        credential_path: str,
        hash_on_feature: str,
        table_name: str,
        ds: str,
        epochs: int,
        batch_size: int,
        chunk_size: int,
        exclude_features: List[str],
        table_sample_percent: float,
        test_split_percent: float,
        skip_small_batches: bool,
    ):
        super().__init__(
            datastore,
            credential_path,
            hash_on_feature,
            table_name,
            ds,
            epochs,
            batch_size,
            chunk_size,
            exclude_features,
            table_sample_percent,
            test_split_percent,
            skip_small_batches,
        )

    def _get_client(self, credential_path: str):
        credentials = service_account.Credentials.from_service_account_file(
            filename=credential_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        return bigquery.Client(credentials=credentials, project=credentials.project_id)

    def _build_train_and_eval_queries(
        self, hash_on_feature: str, ds: str = None
    ) -> str:
        base_query = f"select * from `{self.table_name}` "

        if ds is not None:
            base_query += f"where date(_PARTITIONTIME) = '{ds}' and "
        else:
            base_query += "where "

        # TODO: right now smallest portion of dataset you can get is 1/1000th
        # maybe make this a user input if it causes problems.
        base_query += f"""
        MOD(ABS(FARM_FINGERPRINT(cast(features.{hash_on_feature} as string))), 1000) / 1000
        """
        train_set_query = base_query + f" < {self.train_set_filter};"
        eval_set_query = (
            base_query + f" between {self.train_set_filter} and {self.max_row_filter}"
        )
        return train_set_query, eval_set_query

    def _create_tmp_table(self, train: bool = True) -> QueryJob:
        """
        Create a temp table to read chunks from. Think of this as creating
        a table to then `list_rows` from in bq.
        """
        if train:
            query = self.train_query
        else:
            query = self.eval_query
        return self.client.query(query=query)

    def _create_query_iterator(self, train: bool = True) -> Tuple[int, Generator]:
        if train:
            query_job = self.train_query_job
        else:
            query_job = self.eval_query_job

        logger.info("Generating temp table with following query:")
        logger.info(query_job.query)
        start = time.time()
        # TODO: .result() is blocking. Perhaps we can run it in a
        # background thread since it can be expensive. One positive - subsequent
        # calls to .result() are fast due to caching in the bigquery client.
        tmp_tbl = query_job.result(page_size=self.chunk_size)
        logger.info(f"Temp table generated. Took {round(time.time() - start, 2)}s.")
        return (tmp_tbl.total_rows, tmp_tbl._to_dataframe_iterable(dtypes={}))

    @retry(tries=3, delay=1, backoff=2, logger=logger)
    def adhoc_query(self, query: str) -> pd.DataFrame:
        """Execute adhoc / one-off blocking query."""
        return self.client.query(query).result().to_dataframe()
