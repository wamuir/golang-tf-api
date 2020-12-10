package classifier

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

const tol = 1e-6

func TestModelStats(t *testing.T) {

	q, err := tf.NewTensor([][]float32{[]float32{float32(0.50), float32(0.50)}})
	assert.Nil(t, err)

	p, err := tf.NewTensor([][]float32{[]float32{float32(0.25), float32(0.75)}})
	assert.Nil(t, err)

	m := Model{nil, nil, q}
	s := m.stats(p, 0)
	assert.NotNil(t, s)

	gi, ok := s["gini-impurity"]
	assert.True(t, ok)
	assert.InDelta(t, float32(0.3750000), gi, tol)

	re, ok := s["relative-entropy"]
	assert.True(t, ok)
	assert.InDelta(t, float32(0.1887219), re, tol)

	se, ok := s["shannon-entropy"]
	assert.True(t, ok)
	assert.InDelta(t, float32(0.8112781), se, tol)

	return
}

func TestModelStatsInfty(t *testing.T) {

	a := float32(math.SmallestNonzeroFloat32)
	b := float32(1 - math.SmallestNonzeroFloat32)

	q, err := tf.NewTensor([][]float32{[]float32{a, b}})
	assert.Nil(t, err)

	p, err := tf.NewTensor([][]float32{[]float32{b, a}})
	assert.Nil(t, err)

	m := Model{nil, nil, q}
	s := m.stats(p, 0)
	assert.NotNil(t, s)

	gi, ok := s["gini-impurity"]
	assert.True(t, ok)
	assert.InDelta(t, float32(math.SmallestNonzeroFloat32), gi, tol)

	re, ok := s["relative-entropy"]
	assert.True(t, ok)
	assert.InDelta(t, float32(math.Log2(math.Pow(2, 149))), re, tol)

	se, ok := s["shannon-entropy"]
	assert.True(t, ok)
	assert.InDelta(t, float32(0), se, tol)

	return
}

func TestModelStatsZero(t *testing.T) {

	a := float32(0)
	b := float32(1)

	q, err := tf.NewTensor([][]float32{[]float32{a, b}})
	assert.Nil(t, err)

	p, err := tf.NewTensor([][]float32{[]float32{b, a}})
	assert.Nil(t, err)

	m := Model{nil, nil, q}
	s := m.stats(p, 0)
	assert.NotNil(t, s)

	gi, ok := s["gini-impurity"]
	assert.True(t, ok)
	assert.InDelta(t, float32(math.SmallestNonzeroFloat32), gi, tol)

	re, ok := s["relative-entropy"]
	assert.True(t, ok)
	assert.InDelta(t, float32(math.MaxFloat32), re, tol)

	se, ok := s["shannon-entropy"]
	assert.True(t, ok)
	assert.InDelta(t, float32(0), se, tol)

	return
}

func TestCensor(t *testing.T) {

	assert.Equal(t, float32(math.MaxFloat32), censor(math.MaxFloat64))
	assert.Equal(t, float32(math.MaxFloat32), censor(math.Inf(1)))
	assert.Equal(t, float32(math.SmallestNonzeroFloat32), censor(math.SmallestNonzeroFloat64))
	assert.Equal(t, float32(math.SmallestNonzeroFloat32), censor(math.Inf(-1)))
	assert.Equal(t, float32(math.Pi), censor(math.Pi))
	return
}
