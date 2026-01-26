package pipeline

import (
	"context"
	"encoding/json"
	"errors"
	"testing"
	"time"
)

// toJSON marshals a string to json.RawMessage for tests
func toJSON(s string) json.RawMessage {
	b, _ := json.Marshal(s)
	return b
}

// MockStage implements Stage for testing
type MockStage struct {
	name        string
	executeFunc func(ctx *Context) error
	executed    bool
}

func (s *MockStage) Name() string {
	return s.name
}

func (s *MockStage) Execute(ctx *Context) error {
	s.executed = true
	if s.executeFunc != nil {
		return s.executeFunc(ctx)
	}
	return nil
}

func TestNewPipeline(t *testing.T) {
	stages := []Stage{
		&MockStage{name: "stage1"},
		&MockStage{name: "stage2"},
	}
	timeout := 30 * time.Second

	p := NewPipeline(stages, timeout)

	if p == nil {
		t.Fatal("expected non-nil pipeline")
	}

	if len(p.Stages()) != 2 {
		t.Errorf("expected 2 stages, got %d", len(p.Stages()))
	}

	if p.Timeout() != timeout {
		t.Errorf("expected timeout %v, got %v", timeout, p.Timeout())
	}
}

func TestPipelineExecute_AllStagesRun(t *testing.T) {
	stage1 := &MockStage{name: "stage1"}
	stage2 := &MockStage{name: "stage2"}
	stage3 := &MockStage{name: "stage3"}

	p := NewPipeline([]Stage{stage1, stage2, stage3}, 30*time.Second)

	req := &Request{
		Model:    "gpt-4",
		Messages: []Message{{Role: "user", Content: toJSON("hello")}},
		Format:   FormatChatCompletion,
	}

	pctx, err := p.Execute(context.Background(), req, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !stage1.executed {
		t.Error("stage1 was not executed")
	}
	if !stage2.executed {
		t.Error("stage2 was not executed")
	}
	if !stage3.executed {
		t.Error("stage3 was not executed")
	}

	if pctx.Request != req {
		t.Error("request not properly set in context")
	}
}

func TestPipelineExecute_StopsOnError(t *testing.T) {
	expectedErr := errors.New("stage2 failed")

	stage1 := &MockStage{name: "stage1"}
	stage2 := &MockStage{
		name: "stage2",
		executeFunc: func(ctx *Context) error {
			return expectedErr
		},
	}
	stage3 := &MockStage{name: "stage3"}

	p := NewPipeline([]Stage{stage1, stage2, stage3}, 30*time.Second)

	req := &Request{
		Model:    "gpt-4",
		Messages: []Message{{Role: "user", Content: toJSON("hello")}},
		Format:   FormatChatCompletion,
	}

	pctx, err := p.Execute(context.Background(), req, nil)

	if err == nil {
		t.Fatal("expected error, got nil")
	}

	var pipelineErr *PipelineError
	if !errors.As(err, &pipelineErr) {
		t.Errorf("expected PipelineError, got %T", err)
	}

	if pipelineErr.Stage != "stage2" {
		t.Errorf("expected stage 'stage2', got %q", pipelineErr.Stage)
	}

	if !stage1.executed {
		t.Error("stage1 should have executed")
	}
	if !stage2.executed {
		t.Error("stage2 should have executed")
	}
	if stage3.executed {
		t.Error("stage3 should not have executed")
	}

	if pctx.State.Current() != StateFailed {
		t.Errorf("expected state Failed, got %s", pctx.State.Current())
	}
}

func TestPipelineExecute_SetsEmitter(t *testing.T) {
	emitter := &MockEventEmitter{}
	stage := &MockStage{
		name: "check_emitter",
		executeFunc: func(ctx *Context) error {
			if ctx.Emitter != emitter {
				t.Error("emitter not properly set")
			}
			return nil
		},
	}

	p := NewPipeline([]Stage{stage}, 30*time.Second)

	req := &Request{
		Model:    "gpt-4",
		Messages: []Message{{Role: "user", Content: toJSON("hello")}},
		Format:   FormatChatCompletion,
	}

	_, err := p.Execute(context.Background(), req, emitter)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestPipelineExecute_SetsStateTracker(t *testing.T) {
	stage := &MockStage{
		name: "check_state",
		executeFunc: func(ctx *Context) error {
			if ctx.State == nil {
				t.Error("state tracker not set")
			}
			if ctx.State.Current() != StateReceived {
				t.Errorf("expected initial state Received, got %s", ctx.State.Current())
			}
			return nil
		},
	}

	p := NewPipeline([]Stage{stage}, 30*time.Second)

	req := &Request{
		Model:    "gpt-4",
		Messages: []Message{{Role: "user", Content: toJSON("hello")}},
		Format:   FormatChatCompletion,
	}

	_, err := p.Execute(context.Background(), req, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestPipelineExecute_PassesRequestOptions(t *testing.T) {
	var capturedOpts int

	stage := &MockStage{
		name: "check_opts",
		executeFunc: func(ctx *Context) error {
			capturedOpts = len(ctx.ReqOpts)
			return nil
		},
	}

	p := NewPipeline([]Stage{stage}, 30*time.Second)

	req := &Request{
		Model:    "gpt-4",
		Messages: []Message{{Role: "user", Content: toJSON("hello")}},
		Format:   FormatChatCompletion,
	}

	// Pass two options
	_, err := p.Execute(context.Background(), req, nil, nil, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if capturedOpts != 2 {
		t.Errorf("expected 2 request options, got %d", capturedOpts)
	}
}

func TestPipelineExecute_SetsCancelFunc(t *testing.T) {
	stage := &MockStage{
		name: "check_cancel",
		executeFunc: func(ctx *Context) error {
			if ctx.Cancel == nil {
				t.Error("cancel function not set")
			}
			return nil
		},
	}

	p := NewPipeline([]Stage{stage}, 30*time.Second)

	req := &Request{
		Model:    "gpt-4",
		Messages: []Message{{Role: "user", Content: toJSON("hello")}},
		Format:   FormatChatCompletion,
	}

	_, err := p.Execute(context.Background(), req, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestPipelineExecute_ContextPassedBetweenStages(t *testing.T) {
	stage1 := &MockStage{
		name: "stage1",
		executeFunc: func(ctx *Context) error {
			ctx.UserQuery = "modified by stage1"
			return nil
		},
	}

	stage2 := &MockStage{
		name: "stage2",
		executeFunc: func(ctx *Context) error {
			if ctx.UserQuery != "modified by stage1" {
				t.Errorf("expected UserQuery 'modified by stage1', got %q", ctx.UserQuery)
			}
			return nil
		},
	}

	p := NewPipeline([]Stage{stage1, stage2}, 30*time.Second)

	req := &Request{
		Model:    "gpt-4",
		Messages: []Message{{Role: "user", Content: toJSON("hello")}},
		Format:   FormatChatCompletion,
	}

	_, err := p.Execute(context.Background(), req, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestPipelineExecute_EmptyPipeline(t *testing.T) {
	p := NewPipeline([]Stage{}, 30*time.Second)

	req := &Request{
		Model:    "gpt-4",
		Messages: []Message{{Role: "user", Content: toJSON("hello")}},
		Format:   FormatChatCompletion,
	}

	pctx, err := p.Execute(context.Background(), req, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if pctx.State.Current() != StateReceived {
		t.Errorf("expected state Received, got %s", pctx.State.Current())
	}
}

func TestPipelineExecute_ErrorMetadata(t *testing.T) {
	testErr := errors.New("test error")

	stage := &MockStage{
		name: "failing_stage",
		executeFunc: func(ctx *Context) error {
			return testErr
		},
	}

	p := NewPipeline([]Stage{stage}, 30*time.Second)

	req := &Request{
		Model:    "gpt-4",
		Messages: []Message{{Role: "user", Content: toJSON("hello")}},
		Format:   FormatChatCompletion,
	}

	pctx, _ := p.Execute(context.Background(), req, nil)

	// Check that failed state has metadata
	history := pctx.State.History()
	if len(history) < 1 {
		t.Fatal("expected at least 1 transition (to failed)")
	}

	lastTransition := history[len(history)-1]
	if lastTransition.To != StateFailed {
		t.Errorf("expected last transition to Failed, got %s", lastTransition.To)
	}

	if lastTransition.Metadata == nil {
		t.Fatal("expected metadata on failed transition")
	}

	if lastTransition.Metadata["stage"] != "failing_stage" {
		t.Errorf("expected stage 'failing_stage', got %v", lastTransition.Metadata["stage"])
	}
}
