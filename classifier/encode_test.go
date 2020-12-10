package classifier

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestQuantize(t *testing.T) {

	assert.NotPanics(t, func() { quantize(chars) })
	return
}

func TestInvEquantize(t *testing.T) {

	e := quantize(chars)
	f, ok := invQuantize(e)
	assert.True(t, ok)
	assert.Equal(t, *f, chars)
	return
}
