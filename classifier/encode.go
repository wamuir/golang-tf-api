package classifier

import "strings"

const (
	chars     = " !\"#$%&'()*+,-./0123456789:;<=>?@[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"
	maxLength = 250
)

// Encode input text
func quantize(s string) []int32 {

	a := make([]int32, maxLength)

	// Lowercase and cast as array of runes
	r := []rune(strings.ToLower(s))

	// Truncate array of runes if longer than maximum
	if len(r) > maxLength {
		r = r[0:maxLength]
	}

	// Generate right-padded array of indices, in reverse
	for i, e := range r {
		var p int = len(r) - (i + 1)
		a[p] = int32(strings.Index(chars, string(e)) + 1)
	}

	return a
}

// Decode quantized text
func invQuantize(a []int32) (*string, bool) {

	var r []rune

	if len(a) > maxLength {
		return nil, false
	}

	for i := len(a); i > 0; i-- {
		e := a[i-1]
		if len(r) == 0 && e == 0 {
			continue
		}
		r = append(r, []rune(chars)[e-1])
	}

	s := string(r)

	return &s, true
}
