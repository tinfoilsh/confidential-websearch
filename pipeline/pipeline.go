package pipeline

import (
	"context"

	"github.com/openai/openai-go/v3/option"
)

// Pipeline orchestrates the execution of stages for processing requests
type Pipeline struct {
	stages []Stage
}

// NewPipeline creates a new pipeline with the given stages
func NewPipeline(stages []Stage) *Pipeline {
	return &Pipeline{
		stages: stages,
	}
}

// Execute runs all stages in order, stopping on first error
func (p *Pipeline) Execute(ctx context.Context, req *Request, emitter EventEmitter, reqOpts ...option.RequestOption) (*Context, error) {
	ctx, cancel := context.WithCancel(ctx)

	// Create pipeline context
	pctx := &Context{
		Context: ctx,
		Request: req,
		State:   NewStateTracker(),
		Emitter: emitter,
		ReqOpts: reqOpts,
		Cancel:  cancel,
	}

	// Execute stages
	for _, stage := range p.stages {
		if err := stage.Execute(pctx); err != nil {
			// Cancel context immediately to stop any background work
			cancel()

			// Transition to failed state
			pctx.State.Transition(StateFailed, map[string]any{
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

