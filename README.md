# golang-tf-api

API for a character-level convolutional neural network, using a model
exported from tensorflow. The API binds to port 5000 by default, and
inferences can be obtained by calling `/predict` such as:

```sh
curl -X POST -H 'Accept: application/vnd.api+json' \
     -H 'Content-Type: application/vnd.api+json' \
     -d '{"data":{"type": "descriptions", "attributes":{"raw": "portal gun"}}}' \
     -i 'http://localhost:5000/predict'
```

This returns classes and associated probability estimates, in the form of:

```json
{
    "data": [
        {"id":"string", "type":"string", "meta":{"weight":"float32", "rank":"int"}},
        {"id":"string", "type":"string", "meta":{"weight":"float32", "rank":"int"}},
        ...,
        {"id":"string", "type":"string", "meta":{"weight":"float32", "rank":"int"}},
    ]
    "meta":{
        "gini-impurity":"float32",
        "relative-entropy":"float32",
        "shannon-entropy":"float32"
    }
}
```

With classes in the relevant classlist identified by `id` and sorted in descending
order by corresponding probability (`weight`).
