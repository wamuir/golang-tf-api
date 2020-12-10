package classifier

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"path/filepath"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/wamuir/go-jsonapi-core"
)

// errors
var (
	ErrUnreadableClassFile = fmt.Errorf("unable to read classes")
)

// Model holds the trained tensorflow model and related information
type Model struct {
	*tf.SavedModel                 // tf model composition
	Classes        core.Collection // class list
	Q              *tf.Tensor      // reference probability distribution
}

// Load imports a saved tensorflow classifier
// caller should make a deferred call to Model.Session.Close()
func Load(dir string, graph []string) (*Model, error) {

	// Read in exported TF model
	m, err := tf.LoadSavedModel(dir, graph, nil)
	if err != nil {
		return nil, err
	}

	// Read in exported class list
	c, err := readClasses(dir)
	if err != nil {
		return nil, err
	}

	clf := Model{m, *c, nil}

	// Obtain probabilities for a null prediction
	q, err := clf.classify([]string{""})
	if err != nil {
		return nil, err
	}

	clf.Q = q

	return &clf, nil
}

func readClasses(dir string) (*core.Collection, error) {

	var d core.Document

	b, err := ioutil.ReadFile(filepath.Join(dir, "classes.json"))
	if err != nil {
		return nil, err
	}

	json.Unmarshal(b, &d)

	if err := d.AssertDataType(); err != nil {
		return nil, err
	}

	c, ok := d.Data.(core.Collection)
	if !ok {
		return nil, ErrUnreadableClassFile
	}

	return &c, nil
}
