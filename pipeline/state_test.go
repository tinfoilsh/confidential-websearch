package pipeline

import (
	"testing"
	"time"
)

func TestNewStateTracker(t *testing.T) {
	tracker := NewStateTracker()

	if tracker.Current() != StateReceived {
		t.Errorf("expected initial state %s, got %s", StateReceived, tracker.Current())
	}

	if len(tracker.History()) != 0 {
		t.Errorf("expected empty history, got %d transitions", len(tracker.History()))
	}
}

func TestValidTransitions(t *testing.T) {
	tests := []struct {
		name    string
		from    State
		to      State
		wantErr bool
	}{
		{"received to processing", StateReceived, StateProcessing, false},
		{"received to failed", StateReceived, StateFailed, false},
		{"processing to responding", StateProcessing, StateResponding, false},
		{"processing to failed", StateProcessing, StateFailed, false},
		{"responding to completed", StateResponding, StateCompleted, false},
		{"responding to failed", StateResponding, StateFailed, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tracker := &DefaultStateTracker{
				current:      tt.from,
				history:      []StateTransition{},
				stateStarted: map[State]time.Time{tt.from: time.Now()},
			}

			err := tracker.Transition(tt.to, nil)
			if (err != nil) != tt.wantErr {
				t.Errorf("Transition() error = %v, wantErr %v", err, tt.wantErr)
			}

			if err == nil && tracker.Current() != tt.to {
				t.Errorf("expected state %s after transition, got %s", tt.to, tracker.Current())
			}
		})
	}
}

func TestInvalidTransitions(t *testing.T) {
	tests := []struct {
		name string
		from State
		to   State
	}{
		{"received to completed", StateReceived, StateCompleted},
		{"received to responding", StateReceived, StateResponding},
		{"processing to completed", StateProcessing, StateCompleted},
		{"completed to anything", StateCompleted, StateProcessing},
		{"failed to anything", StateFailed, StateReceived},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tracker := &DefaultStateTracker{
				current:      tt.from,
				history:      []StateTransition{},
				stateStarted: map[State]time.Time{tt.from: time.Now()},
			}

			err := tracker.Transition(tt.to, nil)
			if err == nil {
				t.Errorf("expected error for invalid transition from %s to %s", tt.from, tt.to)
			}

			if tracker.Current() != tt.from {
				t.Errorf("state should remain %s after failed transition, got %s", tt.from, tracker.Current())
			}
		})
	}
}

func TestTransitionHistory(t *testing.T) {
	tracker := NewStateTracker()

	tracker.Transition(StateProcessing, map[string]any{"query": "test"})
	tracker.Transition(StateResponding, nil)
	tracker.Transition(StateCompleted, nil)

	history := tracker.History()
	if len(history) != 3 {
		t.Fatalf("expected 3 transitions, got %d", len(history))
	}

	expectedTransitions := []struct {
		from State
		to   State
	}{
		{StateReceived, StateProcessing},
		{StateProcessing, StateResponding},
		{StateResponding, StateCompleted},
	}

	for i, expected := range expectedTransitions {
		if history[i].From != expected.from || history[i].To != expected.to {
			t.Errorf("transition %d: expected %s->%s, got %s->%s",
				i, expected.from, expected.to, history[i].From, history[i].To)
		}
	}

	// Check metadata was recorded
	if history[0].Metadata["query"] != "test" {
		t.Error("expected metadata to be recorded")
	}
}

func TestTransitionTimestamps(t *testing.T) {
	tracker := NewStateTracker()

	before := time.Now()
	tracker.Transition(StateProcessing, nil)
	after := time.Now()

	history := tracker.History()
	if len(history) != 1 {
		t.Fatal("expected 1 transition")
	}

	ts := history[0].Timestamp
	if ts.Before(before) || ts.After(after) {
		t.Error("timestamp should be between before and after")
	}
}

func TestDuration(t *testing.T) {
	tracker := NewStateTracker()

	// Sleep briefly to ensure measurable duration
	time.Sleep(10 * time.Millisecond)

	tracker.Transition(StateProcessing, nil)

	duration := tracker.Duration(StateReceived)
	if duration < 10*time.Millisecond {
		t.Errorf("expected duration >= 10ms, got %v", duration)
	}
}

func TestDurationCurrentState(t *testing.T) {
	tracker := NewStateTracker()
	tracker.Transition(StateProcessing, nil)

	time.Sleep(10 * time.Millisecond)

	duration := tracker.Duration(StateProcessing)
	if duration < 10*time.Millisecond {
		t.Errorf("expected current state duration >= 10ms, got %v", duration)
	}
}

func TestDurationUnvisitedState(t *testing.T) {
	tracker := NewStateTracker()

	duration := tracker.Duration(StateResponding)
	if duration != 0 {
		t.Errorf("expected 0 duration for unvisited state, got %v", duration)
	}
}

func TestAnyStateCanTransitionToFailed(t *testing.T) {
	states := []State{
		StateReceived,
		StateProcessing,
		StateResponding,
	}

	for _, state := range states {
		t.Run(string(state), func(t *testing.T) {
			tracker := &DefaultStateTracker{
				current:      state,
				history:      []StateTransition{},
				stateStarted: map[State]time.Time{state: time.Now()},
			}

			err := tracker.Transition(StateFailed, map[string]any{"error": "test error"})
			if err != nil {
				t.Errorf("should be able to transition from %s to failed: %v", state, err)
			}
		})
	}
}
