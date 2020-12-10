# golang-tf-api


## About

API for a character-level convolutional neural network, using a model
exported from tensorflow. The API binds to port 5000 by default, and
inferences can be obtained by posting to `/predict`, optionally with
a limit on the number of predicted classes to be returned, such as
`/predict?page[limit]=10`.


## application/json

```sh
curl -XPOST -d '{"input":"portal gun"}' -i http://localhost:5000/predict
```

This returns product service codes and probability estimates, such as:

```json
{
    "classes":[
        {"id":"string","pr":"float32"},
        {"id":"string","pr":"float32"},
        {"id":"string","pr":"float32"}
    ],
    "meta":{
        "gini-impurity":"float32",
        "relative-entropy":"float32",
        "shannon-entropy":"float32"
    }
}
```

With classes in the relevant classlist identified by `id` and sorted in
descending order by corresponding probability (`pr`).



## application/vnd.api+json

The API will negotiate content for [https://jsonapi.org/](JSON API).

```sh
curl -X POST -H 'Accept: application/vnd.api+json' \
     -H 'Content-Type: application/vnd.api+json' \
     -d '{"data":{"type": "descriptions", "attributes":{"raw": "portal gun"}}}' \
     -i 'http://localhost:5000/predict'
```
