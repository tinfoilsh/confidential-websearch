package pipeline

import (
	"context"
	"time"

	"github.com/openai/openai-go/v2/option"
)

// Pipeline orchestrates the execution of stages for processing requests
type Pipeline struct {
	stages  []Stage
	timeout time.Duration
}

// NewPipeline creates a new pipeline with the given stages
func NewPipeline(stages []Stage, timeout time.Duration) *Pipeline {
	return &Pipeline{
		stages:  stages,
		timeout: timeout,
	}
}

// Execute runs all stages in order, stopping on first error
func (p *Pipeline) Execute(ctx context.Context, req *Request, emitter EventEmitter, reqOpts ...option.RequestOption) (*Context, error) {
	// Create context with timeout
	timeoutCtx, cancel := context.WithTimeout(ctx, p.timeout)

	// Create pipeline context
	pctx := &Context{
		Context: timeoutCtx,
		Request: req,
		State:   NewStateTracker(),
		Emitter: emitter,
		ReqOpts: reqOpts,
		Cancel:  cancel,
	}

	// Execute stages
	for _, stage := range p.stages {
		if err := stage.Execute(pctx); err != nil {
			// Transition to failed state
			pctx.State.Transition(StateFailed, map[string]interface{}{
				"stage": stage.Name(),
				"error": err.Error(),
			})

			// Wrap error with stage context
			return pctx, &PipelineError{
				Stage: stage.Name(),
				Err:   err,
			}
		}
	}

	return pctx, nil
}

// Stages returns the pipeline's stages (for testing)
func (p *Pipeline) Stages() []Stage {
	return p.stages
}

// Timeout returns the pipeline's timeout (for testing)
func (p *Pipeline) Timeout() time.Duration {
	return p.timeout
}
