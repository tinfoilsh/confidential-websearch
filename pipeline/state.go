package pipeline

import (
	"fmt"
	"sync"
	"time"
)

// State represents the current phase of request processing
type State string

const (
	StateReceived           State = "received"
	StateAgentStarted       State = "agent_started"
	StateSearchStarted      State = "search_started"
	StateSearchCompleted    State = "search_completed"
	StateAgentCompleted     State = "agent_completed"
	StateResponderStarted   State = "responder_started"
	StateResponderStreaming State = "responder_streaming"
	StateCompleted          State = "completed"
	StateFailed             State = "failed"
)

// validTransitions defines which state transitions are allowed
var validTransitions = map[State][]State{
	StateReceived:           {StateAgentStarted, StateFailed},
	StateAgentStarted:       {StateSearchStarted, StateAgentCompleted, StateFailed},
	StateSearchStarted:      {StateSearchCompleted, StateFailed},
	StateSearchCompleted:    {StateAgentCompleted, StateFailed},
	StateAgentCompleted:     {StateResponderStarted, StateFailed},
	StateResponderStarted:   {StateResponderStreaming, StateCompleted, StateFailed},
	StateResponderStreaming: {StateCompleted, StateFailed},
	StateCompleted:          {},
	StateFailed:             {},
}

// StateTransition records a state change
type StateTransition struct {
	From      State
	To        State
	Timestamp time.Time
	Metadata  map[string]interface{}
}

// StateTracker tracks request lifecycle state
type StateTracker interface {
	Current() State
	Transition(to State, metadata map[string]interface{}) error
	History() []StateTransition
	Duration(state State) time.Duration
}

// DefaultStateTracker is an in-memory state tracker implementation
type DefaultStateTracker struct {
	mu           sync.RWMutex
	current      State
	history      []StateTransition
	stateStarted map[State]time.Time
}

// NewStateTracker creates a new state tracker starting at StateReceived
func NewStateTracker() *DefaultStateTracker {
	now := time.Now()
	return &DefaultStateTracker{
		current: StateReceived,
		history: []StateTransition{},
		stateStarted: map[State]time.Time{
			StateReceived: now,
		},
	}
}

// Current returns the current state
func (t *DefaultStateTracker) Current() State {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.current
}

// Transition moves to a new state if the transition is valid
func (t *DefaultStateTracker) Transition(to State, metadata map[string]interface{}) error {
	t.mu.Lock()
	defer t.mu.Unlock()

	if !t.isValidTransition(t.current, to) {
		return fmt.Errorf("invalid state transition from %s to %s", t.current, to)
	}

	now := time.Now()
	transition := StateTransition{
		From:      t.current,
		To:        to,
		Timestamp: now,
		Metadata:  metadata,
	}

	t.history = append(t.history, transition)
	t.current = to

	if t.stateStarted == nil {
		t.stateStarted = make(map[State]time.Time)
	}
	t.stateStarted[to] = now

	return nil
}

// History returns all state transitions
func (t *DefaultStateTracker) History() []StateTransition {
	t.mu.RLock()
	defer t.mu.RUnlock()

	result := make([]StateTransition, len(t.history))
	copy(result, t.history)
	return result
}

// Duration returns how long was spent in a given state
func (t *DefaultStateTracker) Duration(state State) time.Duration {
	t.mu.RLock()
	defer t.mu.RUnlock()

	started, ok := t.stateStarted[state]
	if !ok {
		return 0
	}

	// Find when this state ended (next transition from this state)
	for _, tr := range t.history {
		if tr.From == state {
			return tr.Timestamp.Sub(started)
		}
	}

	// State is still current or was never transitioned from
	if t.current == state {
		return time.Since(started)
	}

	return 0
}

func (t *DefaultStateTracker) isValidTransition(from, to State) bool {
	validTargets, ok := validTransitions[from]
	if !ok {
		return false
	}

	for _, valid := range validTargets {
		if valid == to {
			return true
		}
	}
	return false
}
