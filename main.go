package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"
	"time"

	"github.com/go-chi/chi"
	"github.com/go-chi/chi/middleware"
	"github.com/wamuir/go-jsonapi-core"
	"github.com/wamuir/golang-tf-api/classifier"
)

// command line flag for model path
var mpath *string = flag.String("path", "models/latest", "path to model directory")

// environment
type env struct {
	*classifier.Model
	stderr *log.Logger
	stdout *log.Logger
}

// Prediction endpoint
func (e *env) predict(w http.ResponseWriter, r *http.Request) {

	var (
		buf     bytes.Buffer
		content interface{}
		raw     string
	)

	start := time.Now()

	// Read only the first 1024 bytes of the request body.
	// Expect HTTP 400 (Bad Request) for larger requests.
	body := http.MaxBytesReader(w, r.Body, 2<<9)

	var limit int
	if p := r.URL.Query().Get("page[limit]"); p != "" {
		l, err := strconv.Atoi(r.URL.Query().Get("page[limit]"))
		if err != nil {
			e.handleError(w, http.StatusBadRequest)
			return
		}
		limit = l
	}

	// Negotiate content
	switch r.Header.Get("Content-Type") {

	case "application/vnd.api+json":

		var request core.Document

		decoder := json.NewDecoder(body)
		if err := decoder.Decode(&request); err != nil {
			e.handleError(w, http.StatusBadRequest)
			return
		}

		if err := request.AssertDataType(); err != nil {
			e.handleError(w, http.StatusBadRequest)
			return
		}

		input, ok := request.Data.(core.Resource)
		if !ok {
			e.handleError(w, http.StatusBadRequest)
			return
		}

		att, ok := input.Attributes["raw"]
		if !ok {
			e.handleError(w, http.StatusBadRequest)
			return
		}

		r, ok := att.(string)
		if !ok {
			e.handleError(w, http.StatusBadRequest)
			return
		}

		raw = r

	default:

		var request struct {
			Input string `json:"input"`
		}

		// Unmarshal body into Request struct
		decoder := json.NewDecoder(body)

		if err := decoder.Decode(&request); err != nil {
			e.handleError(w, http.StatusBadRequest)
			return
		}

		raw = request.Input

	}

	// Obtain predictions on the input
	results, err := e.Predict([]string{raw}, limit)
	if err != nil {
		e.stderr.Println(err)
		e.handleError(w, http.StatusInternalServerError)
		return
	}

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
		a := results[0].Data.(classifier.Result)
		data := make([]map[string]interface{}, len(a))
		for i, r := range a {
			data[i] = map[string]interface{}{
				"id": r.Identifier,
				"pr": r.Meta["association"],
			}
		}

		content = map[string]interface{}{
			"classes": data,
			"meta":    results[0].Meta,
		}
	}

	// Add server timing in header
	duration := fmt.Sprintf("%.4f", time.Since(start).Seconds())
	w.Header().Set("Server-Timing", "total;dur="+duration)

	encoder := json.NewEncoder(&buf)
	if err := encoder.Encode(content); err != nil {
		e.stderr.Println(err)
		e.handleError(w, http.StatusInternalServerError)
		return
	}

	// Write the response
	w.WriteHeader(http.StatusOK)
	if _, err := w.Write(buf.Bytes()); err != nil {
		e.stderr.Println(err)
		return
	}

	return
}

// Handle HTTP errors
func (e *env) handleError(w http.ResponseWriter, code int) {

	var (
		buf     bytes.Buffer
		content interface{}
	)

	content = map[string]string{"Error": http.StatusText(code)}

	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.Header().Set("Content-Type", "application/json")

	encoder := json.NewEncoder(&buf)
	if err := encoder.Encode(content); err != nil {
		e.stderr.Println(err)
		e.handleError(w, http.StatusInternalServerError)
		return
	}

	// Write the response
	w.WriteHeader(code)
	if _, err := w.Write(buf.Bytes()); err != nil {
		e.stderr.Println(err)
		return
	}

	return
}

// Endpoint for health checks by haproxy, etc.
func (e *env) health(w http.ResponseWriter, r *http.Request) {

	var (
		buf     bytes.Buffer
		content interface{}
	)

	content = map[string]bool{"ok": true}

	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.Header().Set("Content-Type", "application/json")

	encoder := json.NewEncoder(&buf)
	if err := encoder.Encode(content); err != nil {
		e.stderr.Println(err)
		e.handleError(w, http.StatusInternalServerError)
		return
	}

	// Write the response
	w.WriteHeader(http.StatusOK)
	if _, err := w.Write(buf.Bytes()); err != nil {
		e.stderr.Println(err)
		return
	}

	return
}

func main() {

	flag.Parse()

	// Set up loggers
	stderr := log.New(os.Stderr, "ERROR: ", log.LstdFlags|log.LUTC)
	stdout := log.New(os.Stdout, "INFO: ", log.LstdFlags|log.LUTC)

	// Read in exported TF model
	m, err := classifier.Load(*mpath, []string{"serve"})
	if err != nil {
		stderr.Fatal(err.Error())
	}
	defer m.Session.Close()

	// Make environment
	e := env{m, stderr, stdout}

	// A righteous mux
	r := chi.NewRouter()
	r.Use(middleware.RequestID)
	r.Use(middleware.RealIP)
	r.Use(middleware.Logger)
	r.Use(middleware.NoCache)
	r.Use(middleware.Recoverer)
	r.Use(middleware.Timeout(5 * time.Second))
	r.Get("/health", e.health)
	r.Post("/predict", e.predict)
	r.NotFound(func(w http.ResponseWriter, r *http.Request) {
		e.handleError(w, 404)
	})
	r.MethodNotAllowed(func(w http.ResponseWriter, r *http.Request) {
		e.handleError(w, 405)
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
