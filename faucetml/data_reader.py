import math
import threading
import time
from typing import Dict, Generator, List, NoReturn, Tuple
import queue

import feast
from google.cloud import bigquery
from google.oauth2 import service_account
from google.cloud.bigquery.job import QueryJob
import pandas as pd
from retry import retry
import torch

from .preprocessing import identify_types
from .preprocessing.normalization import (
    NormalizationParameters,
    sort_features_by_normalization,
)
from .preprocessing.norm_metadata import get_norm_metadata_dict
from .preprocessing.preprocessor_net import Preprocessor
from .preprocessing.sparse_to_dense import PandasSparseToDenseProcessor
from .utils import get_logger


logger = get_logger(__name__)


class DataStore:
    bigquery = "bigquery"


def get_client(
    datastore: str,
    credential_path: str,
    table_name: str,
    ds: str = None,
    epochs: int = 1,
    batch_size: int = 1024,
    chunk_size: int = 1024 * 100,
    exclude_features: List[str] = None,
    table_sample_percent: float = None,
    test_split_percent: float = None,
    skip_small_batches: bool = True,
    faucet_core_url: str = None,
    faucet_batch_serving_url: str = None,
    faucet_project: str = None,
):
    """Get the Faucet client.

    Args:
        datastore: Datastore fetching data from (i.e. "bigquery").
        credential_path: Path to credentials for datastore.
        table_name: Name of table to fetch data from.
        ds: Date for table partition.
        epochs: Number of epochs to train for.
        batch_size: Mini-batch size in training.
        chunk_size: Size of data to fetch from table. Should be >> batch_size
            but small enough to fit in memory.
        exclude_features: Features to exclude in training.
        table_sample_percent: Percent of table to use for training and eval.
        test_split_percent: Percent of table sample to use for eval.
        faucet_core_url: Core URL of feature & preprocessing store. Used when
            features are stored in a remote faucet cluster.
        faucet_batch_serving_url: Serving URL for batch features & preprocessing.
        faucet_project: Namespace in faucet where the features & preprocessing
            specs are stored.

    Returns:
        DataReader object.

    """

    if datastore == DataStore.bigquery:
        return BigQueryReader(
            datastore,
            credential_path,
            table_name,
            ds,
            epochs,
            batch_size,
            chunk_size,
            exclude_features,
            table_sample_percent,
            test_split_percent,
            skip_small_batches,
            faucet_core_url,
            faucet_batch_serving_url,
            faucet_project,
        )
    else:
        raise Exception("Datastore {} not supported".format(kwargs["datastore"]))


def get_online_reader():
    pass


class DataReader:
    def __init__(
        self,
        datastore: str,
        credential_path: str,
        table_name: str,
        ds: str,
        epochs: int,
        batch_size: int,
        chunk_size: int,
        exclude_features: List[str],
        table_sample_percent: float,
        test_split_percent: float,
        skip_small_batches: bool,
        faucet_core_url: str,
        faucet_batch_serving_url: str,
        faucet_project: str,
    ):
        # Intialize clients & set table properties
        self.client = self._get_client(credential_path)
        self.table_name = table_name
        self.exclude_features = exclude_features
        if faucet_core_url is not None:
            assert faucet_batch_serving_url is not None
            assert faucet_project is not None
            self.feast_client = feast.Client(
                core_url=faucet_core_url, serving_url=faucet_batch_serving_url
            )
            self.feast_client.set_project(faucet_project)

        # Figure out sampling + train & test
        self.max_row_filter = table_sample_percent / 100
        self.train_set_filter = self.max_row_filter * (1 - test_split_percent / 100)

        # Generate queries for the training & test set
        self.train_query, self.eval_query = self._build_train_and_eval_queries(ds)

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

        # Preprocessing specifications & preprocessor net
        self.preproc_specifications = None
        self.preprocessor_net = None

    def _get_client(self, credential_path: str):
        raise NotImplementedError

    def _build_train_and_eval_queries(self, ds: str = None) -> str:
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
            "features": pd.DataFrame(chunk_df.features.values.tolist())[
                start_idx:end_idx
            ],
            "labels": pd.DataFrame(chunk_df.labels.values.tolist())[start_idx:end_idx],
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

    def get_batch(self, preprocess: bool = False, eval: bool = False) -> Dict:
        """Get a batch of data from the table.

        Args:
            preprocess: Whether or not to preprocess this batch using the
                preprocessor_net.
            eval: Whether or not this batch is used for evaluation.

        Returns:
            Dict {
                "features": pd.DataFrame or Torch.Tensor,
                "labels": pd.DataFrame or Torch.Tensor,
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

        if preprocess:
            assert (
                self.preprocessor_net is not None
            ), "`preprocessor_net` not defined. Create preprocessing specs before trying to get batch."
            features, labels = self.preprocess_batch(
                batch_dict["features"], batch_dict["labels"]
            )
            batch_dict["features"] = features
            batch_dict["labels"] = labels

        return batch_dict

    def gen_preprocess_specs_and_net(
        self,
        exclude_features: List[str] = [],
        sample_size: int = 10000,
        max_unique_enum_values: int = 20,
        quantile_size: int = 20,
        quantile_k2_threshold: int = 1000.0,
        skip_box_cox: bool = False,
        skip_quantiles: bool = True,
        feature_overrides: Dict[str, str] = {},
        skip_preprocessing: bool = False,
    ) -> Tuple[Dict[str, NormalizationParameters], Preprocessor]:
        """Generate preprocessing specifications for each feature in the
        training data and a PyTorch net that uses those specifications to do
        the preprocessing.

        Args:
            exclude_features: List of features to exclude from training data.
            sample_size: How many rows to use when computing stats (mean, stddev, etc.)
                for features.
            max_unique_enum_values: Max cardinality of feature to be treated as
                enum.
            quantile_size: Number of buckets when discretizing quanitle data.
            quantile_k2_threshold: Boxcox transform K2 threshold.
            skip_box_cox: Skip boxcox transformations.
            skip_quantiles: Skip quantile transformations.
            feature_overrides: Manually override feature type insteas of inferring
                it from the data.
            skip_preprocessing: Whether or not to skip preprocessing.

        Returns:
            Tuple[
                Dict {
                    "feature_a": {...}, "feature_b": {...},
                },
                Preprocessor,
            ]

        """

        data_df = self._get_data_to_create_norm(sample_size)
        self.preproc_specifications = get_norm_metadata_dict(
            data_df=data_df,
            exclude_features=exclude_features,
            feature_overrides=feature_overrides,
            max_unique_enum_values=max_unique_enum_values,
            quantile_size=quantile_size,
            quantile_k2_threshold=quantile_k2_threshold,
            skip_box_cox=skip_box_cox,
            skip_quantiles=skip_quantiles,
            skip_preprocessing=skip_preprocessing,
        )

        preprocessor_net = self._create_preprocessor(self.preproc_specifications)
        return self.preproc_specifications, preprocessor_net

    def get_num_output_features(
        self, preproc_specifications: Dict[str, NormalizationParameters]
    ) -> int:
        """Returns the number of features that will output from the preprocessor.

        Args:
            preproc_specifications: Dict of preprocessing specifications
                generated by running `gen_preprocess_spec_for_features`.

        Returns:
            int: Number of features after preprocessing.
        """
        return sum(
            map(
                lambda np: (
                    len(np.possible_values)
                    if np.feature_type == identify_types.ENUM
                    else 1
                ),
                preproc_specifications.values(),
            )
        )

    def _create_preprocessor(
        self, preproc_specifications: Dict[str, NormalizationParameters]
    ) -> Preprocessor:
        """Creates PyTorch net used to preprocess raw features.

        Args:
            preproc_specifications: Dict of preprocessing specifications
                generated by running `gen_preprocess_spec_for_features`.

        Returns:
            Preprocessor: A PyTorch net that knows how to preprocess a feature.

        """
        assert (
            preproc_specifications is not None
        ), "Run `gen_preprocess_spec_for_features` before `gen_preprocess_net`."

        self.preprocessor_net = Preprocessor(preproc_specifications)
        sorted_features, _ = sort_features_by_normalization(
            self.preprocessor_net.normalization_parameters
        )
        assert sorted_features == self.preprocessor_net.sorted_features
        self.sparse_to_dense = PandasSparseToDenseProcessor(sorted_features)
        return self.preprocessor_net

    def preprocess_batch(
        self, features_df: pd.DataFrame, labels_df: pd.DataFrame = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Converts a sparse training batch to a dense preprocessed batch.

        Args:
            features_df: Dataframe of features.
            labels_df: Dataframe of labels.

        Returns:
            Tuple of dense Torch tensors (training data and labels).
        """
        if labels_df is None:
            labels = None
        else:
            labels = {
                label_name: torch.tensor(labels_df.values, dtype=torch.float)
                for label_name in labels_df.columns
            }
        dense_batch, dense_presence = self.sparse_to_dense.process(features_df)
        return self.preprocessor_net(dense_batch, dense_presence), labels


class BigQueryReader(DataReader):
    def __init__(
        self,
        datastore: str,
        credential_path: str,
        table_name: str,
        ds: str,
        epochs: int,
        batch_size: int,
        chunk_size: int,
        exclude_features: List[str],
        table_sample_percent: float,
        test_split_percent: float,
        skip_small_batches: bool,
        faucet_core_url: str,
        faucet_batch_serving_url: str,
        faucet_project: str,
    ):
        super().__init__(
            datastore,
            credential_path,
            table_name,
            ds,
            epochs,
            batch_size,
            chunk_size,
            exclude_features,
            table_sample_percent,
            test_split_percent,
            skip_small_batches,
            faucet_core_url,
            faucet_batch_serving_url,
            faucet_project,
        )

    def _get_client(self, credential_path: str):
        credentials = service_account.Credentials.from_service_account_file(
            filename=credential_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        return bigquery.Client(credentials=credentials, project=credentials.project_id)

    def _build_train_and_eval_queries(self, ds: str = None) -> str:
        base_query = f"select * from `{self.table_name}` "

        if ds is not None:
            base_query += f"where date(_PARTITIONTIME) = '{ds}' and "
        else:
            base_query += "where "

        # TODO: right now smallest portion of dataset you can get is 1/1000th
        # maybe make this a user input if it causes problems.
        base_query += "MOD(ABS(FARM_FINGERPRINT(cast(hash_on as string))), 1000) / 1000"

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

    def _get_data_to_create_norm(self, sample_size: int) -> pd.DataFrame:
        # assumes training data is already shuffled
        query = f"""
        select
            features.*
        from `{self.table_name}`
        limit {sample_size}
        """
        return self.adhoc_query(query)

    @retry(tries=3, delay=1, backoff=2, logger=logger)
    def adhoc_query(self, query: str) -> pd.DataFrame:
        """Execute adhoc / one-off blocking query."""
        return self.client.query(query).result().to_dataframe()
