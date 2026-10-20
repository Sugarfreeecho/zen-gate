package lane

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"
)

// Unchecking a model on the 模型 page is the user saying "do not spend my quota
// on this one". It used to only remove the model from the picker: the lane kept
// probing it every round and kept offering it as a failover substitute, so a
// model the user never selected still received real upstream requests.
//
// These tests pin the corrected meaning: hidden = not automatically, but still
// servable when a request names it explicitly.

func TestHiddenModelIsNotAFailoverCandidate(t *testing.T) {
	newLane := func(hidden ...string) *Lane {
		l := NewLane()
		l.catalog = []ModelInfo{
			{ID: "a-free", Name: "a", Wire: "chat"},
			{ID: "b-free", Name: "b", Wire: "chat"},
		}
		l.SetHidden(hidden)
		// a is refused, so the only substitute left is b.
		l.availability["a-free"] = ProbeResult{Model: "a-free", State: StateUnavailable,
			At: time.Now().UnixMilli()}
		return l
	}

	// Baseline: without hiding, b is exactly the substitute the lane picks.
	if got := newLane().nextCandidate(Needs{}, map[string]bool{}); got != "b-free" {
		t.Fatalf("baseline candidate = %q, want b-free", got)
	}

	// b hidden: there is no substitute left, and a is refused.
	if got := newLane("b-free").nextCandidate(Needs{}, map[string]bool{}); got != "" {
		t.Fatalf("a hidden model was offered as a substitute: %q", got)
	}
}

func TestHiddenModelsStayOutOfTheCandidatesHint(t *testing.T) {
	l := NewLane()
	l.catalog = []ModelInfo{
		{ID: "a-free", Name: "a", Wire: "chat"},
		{ID: "b-free", Name: "b", Wire: "chat"},
		{ID: "c-free", Name: "c", Wire: "chat"},
	}
	l.SetHidden([]string{"b-free"})

	// The 429 body advertises these as the models the gateway could switch to,
	// so a hidden id must never appear there either.
	got := l.Candidates(Needs{}, "a-free", 5)
	for _, id := range got {
		if id == "b-free" {
			t.Fatalf("hidden model leaked into the candidate hint: %v", got)
		}
	}
	if len(got) != 1 || got[0] != "c-free" {
		t.Fatalf("candidates = %v, want [c-free]", got)
	}
}

func TestProbeTargetsSkipHiddenModels(t *testing.T) {
	l := NewLane()
	l.catalog = []ModelInfo{
		{ID: "a-free", Name: "a", Wire: "chat"},
		{ID: "b-free", Name: "b", Wire: "chat"},
		{ID: "c-free", Name: "c", Wire: "chat"},
	}
	l.SetHidden([]string{"B-FREE"}) // spelling/whitespace must not matter
	l.SetHidden([]string{" b-free "})

	got := l.probeTargets()
	want := []string{"a-free", "c-free"}
	if len(got) != len(want) {
		t.Fatalf("probe targets = %v, want %v", got, want)
	}
	for i, m := range got {
		if m.ID != want[i] {
			t.Fatalf("probe targets = %v, want %v", got, want)
		}
	}
}

// A round where every catalog entry is hidden must not touch the network at
// all — not even the egress probe that normally closes a round.
func TestAllHiddenCatalogProbesNothing(t *testing.T) {
	var hits int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&hits, 1)
		w.Header().Set("content-type", "text/event-stream")
		fmt.Fprint(w, chatSSE(`{"choices":[{"delta":{"content":"pong"}}]}`))
	}))
	defer srv.Close()
	old := UpstreamBase
	UpstreamBase = srv.URL
	defer func() { UpstreamBase = old }()

	l := NewLane()
	l.catalog = []ModelInfo{{ID: "a-free", Name: "a", Wire: "chat"}}
	l.SetHidden([]string{"a-free"})

	l.ProbeRound(context.Background(), true)

	if n := atomic.LoadInt32(&hits); n != 0 {
		t.Fatalf("a hidden model was probed %d time(s), want 0", n)
	}
}

// Hiding is not a revocation: an explicit request still reaches the model.
func TestHiddenModelStillServesAnExplicitRequest(t *testing.T) {
	var seen []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Model string `json:"model"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		seen = append(seen, body.Model)
		w.Header().Set("content-type", "text/event-stream")
		fmt.Fprint(w, chatSSE(
			`{"choices":[{"delta":{"role":"assistant","content":"hi"}}]}`,
			`{"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":2}}`,
		))
	}))
	defer srv.Close()
	old := UpstreamBase
	UpstreamBase = srv.URL
	defer func() { UpstreamBase = old }()

	l := NewLane()
	l.catalog = []ModelInfo{{ID: "a-free", Name: "a", Wire: "chat"}}
	l.SetFailover(false, 0)
	l.SetHidden([]string{"a-free"})

	_, uerr := l.Complete(context.Background(), Request{
		Model:    "a-free",
		Messages: []Message{{Role: RoleUser, Parts: []Part{TextPart{Text: "hi"}}}},
	}, func(Chunk) {})
	if uerr != nil {
		t.Fatalf("explicit request to a hidden model failed: %+v", uerr)
	}
	if len(seen) != 1 || seen[0] != "a-free" {
		t.Fatalf("upstream saw %v, want exactly [a-free]", seen)
	}
}

// A hidden model must also not be reached through a failover that starts from a
// different requested model — the case that produced the reported traffic.
func TestFailoverDoesNotLandOnAHiddenModel(t *testing.T) {
	var seen []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Model string `json:"model"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		seen = append(seen, body.Model)
		if body.Model == "a-free" {
			w.Header().Set("content-type", "application/json")
			w.WriteHeader(http.StatusTooManyRequests)
			fmt.Fprint(w, `{"error":{"message":"FreeUsageLimitError"}}`)
			return
		}
		w.Header().Set("content-type", "text/event-stream")
		fmt.Fprint(w, chatSSE(
			`{"choices":[{"delta":{"role":"assistant","content":"hi"}}]}`,
			`{"choices":[{"delta":{},"finish_reason":"stop"}]}`,
		))
	}))
	defer srv.Close()
	old := UpstreamBase
	UpstreamBase = srv.URL
	defer func() { UpstreamBase = old }()

	l := NewLane()
	l.catalog = []ModelInfo{
		{ID: "a-free", Name: "a", Wire: "chat"},
		{ID: "b-free", Name: "b", Wire: "chat"},
	}
	l.SetFailover(true, 2)
	l.SetHidden([]string{"b-free"})

	_, uerr := l.Complete(context.Background(), Request{
		Model:    "a-free",
		Messages: []Message{{Role: RoleUser, Parts: []Part{TextPart{Text: "hi"}}}},
	}, func(Chunk) {})

	for _, m := range seen {
		if m == "b-free" {
			t.Fatalf("failover reached a hidden model: %v", seen)
		}
	}
	if uerr == nil {
		t.Fatalf("expected the 429 to surface once every substitute was hidden")
	}
}
