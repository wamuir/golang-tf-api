package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"math"
	"net/http"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/go-chi/chi"
	"github.com/go-chi/chi/middleware"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

const (
	chars     = " abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
	maxLength = 256
)

type (
	Class struct {
		Identifier    string            `json:"id,omitempty"`
		Type          string            `json:"type,omitempty"`
		Attributes    map[string]string `json:"attributes,omitempty"`
		Meta          *Meta             `json:"meta,omitempty"`
		Relationships map[string]Result `json:"relationships,omitempty"`
	}
	Meta struct {
		Probability float32 `json:"association,omitempty"`
	}
	Model struct {
		*tf.SavedModel
	}
	Request struct {
		Input string `json:"input"`
	}
	Result struct {
		Data []Class            `json:"data"`
		Meta map[string]float32 `json:"meta,omitempty"`
	}
)

var (
	classifier *Model
	classList  []Class
	q          *tf.Tensor
	stderr     *log.Logger
	stdout     *log.Logger
)

func (r Result) Len() int {
	return len(r.Data)
}

func (r Result) Less(i, j int) bool {
	return r.Data[i].Meta.Probability > r.Data[j].Meta.Probability
}

func (r Result) Swap(i, j int) {
	r.Data[i], r.Data[j] = r.Data[j], r.Data[i]
}

// Obtain probability estimates from TF model
func (m Model) classify(input []string) (*tf.Tensor, error) {

	// Encode the string
	encoded := make([][maxLength]float32, len(input))
	for k, v := range input {
		encoded[k] = encode(v)
	}

	// Convert to a tensor
	inputTensor, err := tf.NewTensor(encoded)
	if err != nil {
		return new(tf.Tensor), err
	}

	// Generate inferences from the model
	result, err := m.Session.Run(
		map[tf.Output]*tf.Tensor{
			m.Graph.Operation("inputLayer").Output(0): inputTensor,
		},
		[]tf.Output{
			m.Graph.Operation("outputLayer/Softmax").Output(0),
		},
		nil,
	)
	if err != nil {
		return new(tf.Tensor), err
	}

	return result[0], nil
}

// Gather results set
func (m Model) predict(input []string) ([]Result, error) {

	var results []Result = make([]Result, len(input))

	// Obtain probabilities
	predictions, err := m.classify(input)
	if err != nil {
		return results, err
	}

	// Iterate over each prediction (i.e., if bulk)
	for i := 0; i < int(predictions.Shape()[0]); i += 1 {

		// Result for the result set, including meta stats
		var result Result = Result{
			Data: make([]Class, len(classList)),
			Meta: m.stats(predictions, i),
		}

		// Include class identifiers with probability estimates
		for j, p := range predictions.Value().([][]float32)[i] {
			var class = Class{
				Type:       "product-service-codes",
				Identifier: classList[j].Identifier,
				Meta:       &Meta{Probability: p},
			}
			result.Data[j] = class
		}

		// Sort the classes in place, ordering by probability desc
		sort.Sort(result)

		// Add to results set
		results[i] = result
	}

	return results, nil
}

// Calculate meta statistics
func (m Model) stats(p *tf.Tensor, idx int) map[string]float32 {

	var gi, re, se float64

	// Note: TF will return 32-bit floats
	p32 := p.Value().([][]float32)[idx]
	q32 := q.Value().([][]float32)[0]

	// Note: math.* operates on 64-bit floats
	for i := range p32 {
		var pk float64 = float64(p32[i]) + math.SmallestNonzeroFloat64
		var qk float64 = float64(q32[i]) + math.SmallestNonzeroFloat64
		gi += math.Pow(pk, 2)
		re += pk * math.Log2(pk/qk)
		se += pk * math.Log2(pk)
	}

	// 32-bit precision is more than sufficient for API
	meta := map[string]float32{
		"gini-impurity":    float32(1.00 - gi),
		"relative-entropy": float32(re),
		"shannon-entropy":  float32(-se),
	}

	return meta
}

// Encode input text
func encode(s string) [maxLength]float32 {

	var array [maxLength]float32

	// Lowercase and cast as array of runes
	runes := []rune(strings.ToLower(s))

	// Truncate array of runes if longer than maximum
	if len(runes) > maxLength {
		runes = runes[0:maxLength]
	}

	// Generate right-padded array of indices, in reverse
	for i, r := range runes {
		var pos int = len(runes) - (i + 1)
		array[pos] = float32(strings.Index(chars, string(r)) + 1)
	}

	return array
}

// Prediction endpoint
func predict(w http.ResponseWriter, r *http.Request) {

	var (
		content interface{}
		request Request
	)

	start := time.Now()

	// Read only the first 1024 bytes of the request body.
	// Expect HTTP 400 (Bad Request) for larger requests.
	body := http.MaxBytesReader(w, r.Body, 2<<9)

	// Unmarshal body into Request struct
	decoder := json.NewDecoder(body)
	err := decoder.Decode(&request)
	if err != nil {
		handleError(w, http.StatusBadRequest)
		return
	}

	// Obtain predictions on the input
	results, err := classifier.predict([]string{request.Input})
	if err != nil {
		handleError(w, http.StatusInternalServerError)
		return
	}

	// Negotiate content
	switch r.Header.Get("Accept") {

	case "application/vnd.api+json":

		// Declare jsonapi mime in header
		w.Header().Set("X-Content-Type-Options", "nosniff")
		w.Header().Set("Content-Type", "application/vnd.api+json")

		// Return results object
		// JSON:API 1.0: https://jsonapi.org/
		content = results[0]

	default:
		// Declare application/json mime in header
		w.Header().Set("X-Content-Type-Options", "nosniff")
		w.Header().Set("Content-Type", "application/json")

		// Return results object
		// {"classes":[{"id":string,"pr":float32},...]}
		data := make([]map[string]interface{}, len(results[0].Data))
		for i, class := range results[0].Data {
			data[i] = map[string]interface{}{
				"id": class.Identifier,
				"pr": class.Meta.Probability,
			}
		}
		content = map[string]interface{}{"classes": data, "meta": results[0].Meta}
	}

	// Add server timing in header
	duration := fmt.Sprintf("%.4f", time.Since(start).Seconds())
	w.Header().Set("Server-Timing", "total;dur="+duration)

	// Write the response
	w.WriteHeader(http.StatusOK)
	encoder := json.NewEncoder(w)
	encoder.Encode(content)
	return
}

// Handle HTTP errors
func handleError(w http.ResponseWriter, code int) {

	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.Header().Set("Content-Type", "application/json")

	w.WriteHeader(code)
	encoder := json.NewEncoder(w)
	encoder.SetIndent("", "\t")
	encoder.Encode(map[string]string{"Error": http.StatusText(code)})
	return
}

// Endpoint for health checks by haproxy, etc.
func health(w http.ResponseWriter, r *http.Request) {

	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.Header().Set("Content-Type", "application/json")

	w.WriteHeader(http.StatusOK)
	encoder := json.NewEncoder(w)
	encoder.SetIndent("", "\t")
	encoder.Encode(map[string]bool{"ok": true})
	return
}

func main() {

	// Set up loggers
	stderr = log.New(os.Stderr, "ERROR: ", log.LstdFlags|log.LUTC)
	stdout = log.New(os.Stdout, "INFO: ", log.LstdFlags|log.LUTC)

	// Read in list of classes for model
	file, err := os.Open("charCNN/classes.json")
	if err != nil {
		stderr.Fatal(err.Error())
		return
	}

	jsonBytes, err := ioutil.ReadAll(file)
	if err != nil {
		stderr.Fatal(err.Error())
		return
	}
	file.Close()

	var document struct {
		Data []Class                `json:"data"`
		Meta map[string]interface{} `json:"meta,omitempty"`
	}
	json.Unmarshal(jsonBytes, &document)
	classList = document.Data

	// Read in exported TF model
	c, err := tf.LoadSavedModel("charCNN", []string{"Graph"}, nil)
	if err != nil {
		stderr.Fatal(err.Error())
		return
	}
	defer c.Session.Close()

	classifier = &Model{c}

	// Obtain probabilities for a null prediction
	q, err = classifier.classify([]string{""})
	if err != nil {
		stderr.Fatal(err.Error())
		return
	}

	// A righteous mux
	r := chi.NewRouter()
	r.Use(middleware.RequestID)
	r.Use(middleware.RealIP)
	r.Use(middleware.Logger)
	r.Use(middleware.NoCache)
	r.Use(middleware.Recoverer)
	r.Use(middleware.Timeout(5 * time.Second))
	r.Get("/health", health)
	r.Post("/predict", predict)
	r.NotFound(func(w http.ResponseWriter, r *http.Request) {
		handleError(w, 404)
	})
	r.MethodNotAllowed(func(w http.ResponseWriter, r *http.Request) {
		handleError(w, 405)
	})

	// Start http server
	server := &http.Server{
		Addr:         ":5000",
		Handler:      r,
		ErrorLog:     stderr,
		ReadTimeout:  5 * time.Second,
		WriteTimeout: 10 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	stderr.Fatal(server.ListenAndServe())
	return
}
