package classifier

import (
	"sort"
	"time"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/wamuir/go-jsonapi-core"
)

const (
	inLayer  = "serving_default_inputLayer"
	outLayer = "StatefulPartitionedCall"
)

// Result is returned by a model prediction
type Result []core.Resource

func (r Result) Len() int {
	return len(r)
}

func (r Result) Less(i, j int) bool {
	return r[i].Meta["association"].(float32) > r[j].Meta["association"].(float32)
}

func (r Result) Swap(i, j int) {
	r[i], r[j] = r[j], r[i]
}

// Obtain probability estimates from tf model
func (m Model) classify(input []string) (*tf.Tensor, error) {

	// Encode the string
	encoded := make([][]int32, len(input))
	for k, v := range input {
		encoded[k] = quantize(v)
	}

	// Convert to a tensor
	inputTensor, err := tf.NewTensor(encoded)
	if err != nil {
		return nil, err
	}

	// Generate inferences from the model
	result, err := m.Session.Run(
		map[tf.Output]*tf.Tensor{
			m.Graph.Operation(inLayer).Output(0): inputTensor,
		},
		[]tf.Output{
			m.Graph.Operation(outLayer).Output(0),
		},
		nil,
	)
	if err != nil {
		return nil, err
	}

	return result[0], nil
}

// Predict runs classifier on input and builds documents
func (m Model) Predict(input []string, limit int) ([]core.Document, error) {

	start := time.Now()

	documents := make([]core.Document, len(input))

	// Obtain probabilities
	predictions, err := m.classify(input)
	if err != nil {
		return documents, err
	}

	// Iterate over each prediction (i.e., if bulk)
	for i := 0; i < int(predictions.Shape()[0]); i++ {

		var result Result = make([]core.Resource, len(m.Classes))

		// Include class identifiers with probability estimates
		for j, p := range predictions.Value().([][]float32)[i] {
			var class = core.Resource{
				Type:       m.Classes[i].Type,
				Identifier: m.Classes[j].Identifier,
				Meta:       map[string]interface{}{"association": p},
			}
			result[j] = class
		}

		// Sort results
		sort.Stable(result)

		// Limit results
		if limit > 0 && limit < len(result) {
			result = result[:limit]
		}

		// Rank results
		for k := range result {
			result[k].Meta["rank"] = k + 1
		}

		// Add result to set
		documents[i] = core.Document{
			Data: result,
			Meta: m.stats(predictions, i),
		}

		// Add time took to document meta
		documents[i].Meta["took"] = time.Now().Sub(start).Milliseconds()
	}

	return documents, nil
}
