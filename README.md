

# Faucet ML

Faucet ML is a Python package that enables high speed mini-batch data reading & preprocessing from common data warehouses for machine learning model training.

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


### More about Faucet
Many training datasets are too large to fit in memory, but model training would benefit from using all of the training data. Naively issuing 1 query per mini-batch of data is unnecessarily expensive due round-trip network costs. Faucet is a library that solves these issues by:
* Fetching large "chunks" of data in non-blocking background threads
	* where chunks are much larger than mini-batches, but still fit in memory
* Caching  chunks locally
* Returning mini-batches from cached chunks in O(1) time



### Examples
See [examples/](https://github.com/econti/faucetml/tree/master/examples) for detailed ipython notebook examples on how to use Faucet.

```
# initialize the client
batch_cli = get_batch_reader(
    datastore="bigquery",
    credential_path="bq_creds.json",
    table_name="my_training_table",
    ds="2020-01-20",
    epochs=2,
    batch_size=1024
    chunk_size=1024 * 10000,
    test_split_percent=20,
)
```

```
# train & test
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
- [ ] Add feature imputation & preprocessing to data reading
- [ ] Support reading features from [Feast](https://github.com/gojek/feast)

Suggestions for other features? Open an issue and let us know.
