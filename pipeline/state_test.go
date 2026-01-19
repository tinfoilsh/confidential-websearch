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
		name     string
		from     State
		to       State
		wantErr  bool
	}{
		{"received to agent_started", StateReceived, StateAgentStarted, false},
		{"received to failed", StateReceived, StateFailed, false},
		{"agent_started to search_started", StateAgentStarted, StateSearchStarted, false},
		{"agent_started to agent_completed", StateAgentStarted, StateAgentCompleted, false},
		{"search_started to search_completed", StateSearchStarted, StateSearchCompleted, false},
		{"search_completed to agent_completed", StateSearchCompleted, StateAgentCompleted, false},
		{"agent_completed to responder_started", StateAgentCompleted, StateResponderStarted, false},
		{"responder_started to streaming", StateResponderStarted, StateResponderStreaming, false},
		{"responder_started to completed", StateResponderStarted, StateCompleted, false},
		{"streaming to completed", StateResponderStreaming, StateCompleted, false},
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
		{"received to responder_started", StateReceived, StateResponderStarted},
		{"agent_started to completed", StateAgentStarted, StateCompleted},
		{"completed to anything", StateCompleted, StateAgentStarted},
		{"failed to anything", StateFailed, StateReceived},
		{"search_started to agent_started", StateSearchStarted, StateAgentStarted},
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

	tracker.Transition(StateAgentStarted, map[string]interface{}{"query": "test"})
	tracker.Transition(StateAgentCompleted, nil)
	tracker.Transition(StateResponderStarted, nil)

	history := tracker.History()
	if len(history) != 3 {
		t.Fatalf("expected 3 transitions, got %d", len(history))
	}

	expectedTransitions := []struct {
		from State
		to   State
	}{
		{StateReceived, StateAgentStarted},
		{StateAgentStarted, StateAgentCompleted},
		{StateAgentCompleted, StateResponderStarted},
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
	tracker.Transition(StateAgentStarted, nil)
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

	tracker.Transition(StateAgentStarted, nil)

	duration := tracker.Duration(StateReceived)
	if duration < 10*time.Millisecond {
		t.Errorf("expected duration >= 10ms, got %v", duration)
	}
}

func TestDurationCurrentState(t *testing.T) {
	tracker := NewStateTracker()
	tracker.Transition(StateAgentStarted, nil)

	time.Sleep(10 * time.Millisecond)

	duration := tracker.Duration(StateAgentStarted)
	if duration < 10*time.Millisecond {
		t.Errorf("expected current state duration >= 10ms, got %v", duration)
	}
}

func TestDurationUnvisitedState(t *testing.T) {
	tracker := NewStateTracker()

	duration := tracker.Duration(StateSearchStarted)
	if duration != 0 {
		t.Errorf("expected 0 duration for unvisited state, got %v", duration)
	}
}

func TestAnyStateCanTransitionToFailed(t *testing.T) {
	states := []State{
		StateReceived,
		StateAgentStarted,
		StateSearchStarted,
		StateSearchCompleted,
		StateAgentCompleted,
		StateResponderStarted,
		StateResponderStreaming,
	}

	for _, state := range states {
		t.Run(string(state), func(t *testing.T) {
			tracker := &DefaultStateTracker{
				current:      state,
				history:      []StateTransition{},
				stateStarted: map[State]time.Time{state: time.Now()},
			}

			err := tracker.Transition(StateFailed, map[string]interface{}{"error": "test error"})
			if err != nil {
				t.Errorf("should be able to transition from %s to failed: %v", state, err)
			}
		})
	}
}
