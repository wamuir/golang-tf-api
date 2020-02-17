# golang-tf-api

API for a character-level convolutional neural network, using a model
exported from tensorflow. The API binds to port 5000 by default, and
inferences can be obtained by calling `/predict` such as:

```sh
curl -XPOST -d'{"input":"portal gun"}' http://localhost:5000/predict
```

This returns product service codes and probability estimates, such as:

```json
{
    "classes":[
        {"id":"string","pr":"float32"},
        {"id":"string","pr":"float32"},
        ...,
        {"id":"string","pr":"float32"}
    ],
    "meta":{
        "gini-impurity":"float32",
        "relative-entropy":"float32",
        "shannon-entropy":"float32"
    }
}
```

With classes in the product service code taxonomy identified by `id`
and sorted in descending order by corresponding probability (`pr`).
