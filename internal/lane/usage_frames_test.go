package lane

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// A streaming upstream that repeats its usage block on every chunk reports
// running totals, not deltas. Summing the repeats multiplies the prompt by the
// number of chunks: one install logged 25,149,054 input tokens for a turn whose
// real context was ~76k, and the client then compacted on every request because
// the number sat far past its 180k threshold.
//
// These tests pin the accounting to the last (largest) value per field and keep
// the frame count as evidence.

func TestRepeatedUsageFramesAreNotSummed(t *testing.T) {
	const prompt = 75689
	var b strings.Builder
	for i := 1; i <= 60; i++ {
		fmt.Fprintf(&b, "data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}],\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d}}\n\n",
			prompt, i*14)
	}
	b.WriteString("data: [DONE]\n\n")

	u, err := readSSE(strings.NewReader(b.String()), func([]byte) error { return nil })
	if err != nil {
		t.Fatalf("readSSE: %v", err)
	}
	if u == nil {
		t.Fatal("no usage decoded")
	}
	if u.Input != prompt {
		t.Errorf("input = %d, want %d (the repeated block is a total, not a delta); summing would give %d",
			u.Input, prompt, prompt*60)
	}
	if u.Output != 60*14 {
		t.Errorf("output = %d, want %d (cumulative completion_tokens, last one wins)", u.Output, 60*14)
	}
	if u.Frames != 60 {
		t.Errorf("frames = %d, want 60 — the count is what makes this visible in the access log", u.Frames)
	}
}

// The healthy shape must not change: one terminal usage frame, one reading.
func TestSingleTerminalUsageIsUnchanged(t *testing.T) {
	sse := "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1234,\"completion_tokens\":56}}\n\n" +
		"data: [DONE]\n\n"

	u, err := readSSE(strings.NewReader(sse), func([]byte) error { return nil })
	if err != nil {
		t.Fatalf("readSSE: %v", err)
	}
	if u == nil || u.Input != 1234 || u.Output != 56 {
		t.Fatalf("usage = %+v, want input 1234 / output 56", u)
	}
	if u.Frames != 1 {
		t.Errorf("frames = %d, want 1", u.Frames)
	}
}

// Anthropic's message_delta also carries a cumulative output_tokens, so the
// same rule applies on that wire.
func TestAnthropicCumulativeDeltasAreNotSummed(t *testing.T) {
	sse := "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":1000,\"output_tokens\":5}}}\n\n" +
		"data: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":400}}\n\n" +
		"data: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":900}}\n\n" +
		"data: [DONE]\n\n"

	u, err := readSSE(strings.NewReader(sse), func([]byte) error { return nil })
	if err != nil {
		t.Fatalf("readSSE: %v", err)
	}
	if u.Input != 1000 {
		t.Errorf("input = %d, want 1000 (the prompt is stated once)", u.Input)
	}
	if u.Output != 900 {
		t.Errorf("output = %d, want 900 (cumulative, so the last delta is the total)", u.Output)
	}
	if u.Frames != 3 {
		t.Errorf("frames = %d, want 3", u.Frames)
	}
}

// Absorb is per field: a frame that only carries one number must not wipe the
// others, and a smaller late value must not shrink the total.
func TestAbsorbKeepsTheLargerValuePerField(t *testing.T) {
	u := &Usage{Input: 100, Output: 10, CacheRead: 7}
	u.Absorb(&Usage{Input: 100, Output: 42}) // output grew, input unchanged
	u.Absorb(&Usage{Input: 90, Output: 3})   // a smaller repeat changes nothing
	u.Absorb(nil)                            // must not panic
	if u.Input != 100 || u.Output != 42 || u.CacheRead != 7 {
		t.Fatalf("usage = %+v, want input 100 / output 42 / cacheRead 7", u)
	}
}

// Merge stays additive: a truncated turn's continuation is a second real
// request, and its prompt was genuinely paid twice.
func TestMergeStillAddsSeparateSegments(t *testing.T) {
	u := &Usage{Input: 70, Output: 5, Frames: 1}
	u.Merge(&Usage{Input: 72, Output: 6, Frames: 1})
	if u.Input != 142 || u.Output != 11 {
		t.Fatalf("usage = %+v, want the two segments summed", u)
	}
	if u.Frames != 1 {
		t.Fatalf("frames = %d, want 1: it is a per-response maximum, not a sum", u.Frames)
	}
}

// End to end through the lane: what the client is told must be the real prompt,
// not prompt x chunks.
func TestCompleteReportsRepeatedUsageOnce(t *testing.T) {
	const prompt = 5000
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("content-type", "text/event-stream")
		// 30 content chunks, each repeating the full running usage block.
		for i := 1; i <= 30; i++ {
			fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}],\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d}}\n\n", prompt, i*3)
		}
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":5000,\"completion_tokens\":90}}\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()
	old := UpstreamBase
	UpstreamBase = srv.URL
	defer func() { UpstreamBase = old }()

	l := NewLane()
	l.catalog = []ModelInfo{{ID: "a-free", Name: "a", Wire: "chat"}}
	l.SetFailover(false, 0)

	out, uerr := l.Complete(context.Background(), Request{
		Model:    "a-free",
		Messages: []Message{{Role: RoleUser, Parts: []Part{TextPart{Text: "hi"}}}},
	}, func(Chunk) {})
	if uerr != nil {
		t.Fatalf("complete: %+v", uerr)
	}
	if out.Usage.Input != prompt {
		t.Fatalf("reported input = %d, want %d; summing 30 frames would report %d",
			out.Usage.Input, prompt, prompt*30)
	}
}
