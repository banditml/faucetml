
# <img src="https://webstockreview.net/images/faucet-clipart-cartoon-3.png" width="50">         Faucet ML

Faucet ML is a Python package that enables high speed mini-batch data reading from common data warehouses for machine learning model training.

Faucet ML is designed for cases where:
* Datasets are too large to fit into memory
* Model training requires mini-batches of data (SGD based algorithms)

### Installation
```
pip install faucetml
```

### Supported data warehouses
- [x] Google BigQuery
- [ ] Snowflake (soon)
- [ ] Amazon Redshift (soon)

Suggestions for other DBs to support? Open an issue and let us know.


### More about Faucet
Many training datasets are too large to fit in memory, but model training would benefit from using all of the training data. Naively issuing 1 query per mini-batch of data is unnecessarily expensive due round-trip network costs. Faucet is a library that solves these issues by:
* Fetching large "chunks" of data in non-blocking background threads
	* where chunks are much larger than mini-batches, but still fit in memory
* Caching  chunks locally
* Returning mini-batches from cached chunks in O(1) time



### Examples

Using Faucet is meant to be simple and painless.

#### BigQuery

Faucet takes in a BigQuery table with the following schema:
```
features <STRUCT>
labels <STRUCT>
```
For example:
```
|                      features                    |     labels     |
|--------------------------------------------------|----------------|
| {"age": 16, "ctr": 0.02, , "noise": 341293, ...} | {"clicked": 0} |
```

Initialize the data reader:
```
from faucetml.data_reader import get_data_reader

data_reader = get_data_reader(
    datastore="bigquery",
    credential_path="path/to/bigquery/creds.json",
    hash_on_feature="noise", # feature used to hash for random sampling
    table_name="project.dataset.training_table",
    ds="2020-01-21",
    epochs=2,
    batch_size=1024,
    chunk_size=1024 * 100,
    exclude_features=["noise"],
    table_sample_percent=100,
    test_split_percent=20,
    skip_small_batches=False,
)
```

Start reading data and training:
```
for epoch in range(2):

    # training loop
    data_reader.prep_for_epoch()
    batch = data_reader.get_batch()
    while batch is not None:
        train(batch)
        batch = data_reader.get_batch()

    # evaluation loop
    data_reader.prep_for_eval()
    batch = data_reader.get_batch(eval=True)
    while batch is not None:
        test(batch)
        batch = data_reader.get_batch(eval=True)
```


### Future features
- [ ] Support more data warehouses
- [ ] Add preprocessing to data reading
- [ ] Support reading features from Feast

Suggestions for other features? Open an issue and let us know.
