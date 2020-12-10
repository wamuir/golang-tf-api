package classifier

import (
	"math"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

// Calculate meta statistics
func (m Model) stats(p *tf.Tensor, idx int) map[string]interface{} {

	var gi, re, se float64

	// tf running on 32-bit precision
	p32 := p.Value().([][]float32)[idx]
	q32 := m.Q.Value().([][]float32)[0]

	// golang math package operates on 64-bit floats
	for i := range p32 {
		var pk float64 = float64(p32[i]) + math.SmallestNonzeroFloat64
		var qk float64 = float64(q32[i]) + math.SmallestNonzeroFloat64

		gi += math.Pow(pk, 2)
		re += pk * math.Log2(pk/qk)
		se += pk * math.Log2(pk)
	}

	// 32-bit precision is more than sufficient for API
	meta := map[string]interface{}{
		"gini-impurity":    censor(1.00 - gi),
		"relative-entropy": censor(re),
		"shannon-entropy":  censor(-se),
	}

	return meta
}

// recast as single precision, censor to positive real numbers
func censor(f64 float64) float32 {

	switch {

	case f64 > float64(math.MaxFloat32):
		fallthrough

	case math.IsInf(f64, 1):
		return math.MaxFloat32

	case f64 < float64(math.SmallestNonzeroFloat32):
		fallthrough

	case math.IsInf(f64, -1):
		return math.SmallestNonzeroFloat32

	default:
		return float32(f64)

	}

}
