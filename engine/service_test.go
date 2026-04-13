package engine

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	openai "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/responses"

	"github.com/tinfoilsh/confidential-websearch/fetch"
	"github.com/tinfoilsh/confidential-websearch/pipeline"
	"github.com/tinfoilsh/confidential-websearch/safeguard"
	"github.com/tinfoilsh/confidential-websearch/search"
)

type fakeResponsesClient struct {
	responses []*responses.Response
	streams   []ResponseStream
	params    []responses.ResponseNewParams
}

func (f *fakeResponsesClient) New(ctx context.Context, body responses.ResponseNewParams, opts ...option.RequestOption) (*responses.Response, error) {
	f.params = append(f.params, body)
	resp := f.responses[0]
	f.responses = f.responses[1:]
	return resp, nil
}

func (f *fakeResponsesClient) NewStreaming(ctx context.Context, body responses.ResponseNewParams, opts ...option.RequestOption) ResponseStream {
	f.params = append(f.params, body)
	stream := f.streams[0]
	f.streams = f.streams[1:]
	return stream
}

type fakeStream struct {
	events []responses.ResponseStreamEventUnion
	index  int
}

func (s *fakeStream) Next() bool {
	if s.index >= len(s.events) {
		return false
	}
	s.index++
	return true
}

func (s *fakeStream) Current() responses.ResponseStreamEventUnion {
	return s.events[s.index-1]
}

func (s *fakeStream) Err() error { return nil }

type fakeSearcher struct {
	results map[string][]search.Result
	err     error
}

func (s *fakeSearcher) Search(ctx context.Context, query string, opts search.Options) ([]search.Result, error) {
	if s.err != nil {
		return nil, s.err
	}
	return s.results[query], nil
}

func (s *fakeSearcher) Name() string { return "fake" }

type fakeFetcher struct {
	pages map[string]fetch.FetchedPage
}

func (f *fakeFetcher) FetchURLs(ctx context.Context, urls []string) []fetch.FetchedPage {
	var pages []fetch.FetchedPage
	for _, url := range urls {
		if page, ok := f.pages[url]; ok {
			pages = append(pages, page)
		}
	}
	return pages
}

type fakeSafeguard struct {
	blocked map[string]string
	errs    map[string]error
}

func (f *fakeSafeguard) Check(ctx context.Context, policy, content string) (*safeguard.CheckResult, error) {
	if err, ok := f.errs[content]; ok {
		return nil, err
	}
	if reason, ok := f.blocked[content]; ok {
		return &safeguard.CheckResult{Violation: true, Rationale: reason}, nil
	}
	return &safeguard.CheckResult{}, nil
}

type captureEmitter struct {
	annotations []pipeline.Annotation
	chunks      []string
}

func (e *captureEmitter) EmitSearchCall(id, status, query, reason string, created int64, model string) error {
	return nil
}
func (e *captureEmitter) EmitFetchCall(id, status, url string, created int64, model string) error {
	return nil
}
func (e *captureEmitter) EmitMetadata(id string, created int64, model string, annotations []pipeline.Annotation) error {
	e.annotations = annotations
	return nil
}
func (e *captureEmitter) EmitChunk(data []byte) error {
	e.chunks = append(e.chunks, string(data))
	return nil
}
func (e *captureEmitter) EmitError(err error) error { return nil }
func (e *captureEmitter) EmitDone(id string, created int64, model string, usage openai.CompletionUsage) error {
	return nil
}
func (e *captureEmitter) EmitResponseStart() error { return nil }
func (e *captureEmitter) EmitMessageStart(itemID string) error {
	return nil
}
func (e *captureEmitter) EmitMessageEnd(text string, annotations []pipeline.Annotation) error {
	return nil
}

type serializingEmitter struct {
	active        atomic.Int32
	maxConcurrent atomic.Int32
}

func (e *serializingEmitter) enter() {
	current := e.active.Add(1)
	for {
		max := e.maxConcurrent.Load()
		if current <= max || e.maxConcurrent.CompareAndSwap(max, current) {
			break
		}
	}
	time.Sleep(10 * time.Millisecond)
	e.active.Add(-1)
}

func (e *serializingEmitter) EmitSearchCall(id, status, query, reason string, created int64, model string) error {
	e.enter()
	return nil
}

func (e *serializingEmitter) EmitFetchCall(id, status, url string, created int64, model string) error {
	e.enter()
	return nil
}

func (e *serializingEmitter) EmitMetadata(id string, created int64, model string, annotations []pipeline.Annotation) error {
	return nil
}
func (e *serializingEmitter) EmitChunk(data []byte) error { return nil }
func (e *serializingEmitter) EmitError(err error) error   { return nil }
func (e *serializingEmitter) EmitDone(id string, created int64, model string, usage openai.CompletionUsage) error {
	return nil
}
func (e *serializingEmitter) EmitResponseStart() error             { return nil }
func (e *serializingEmitter) EmitMessageStart(itemID string) error { return nil }
func (e *serializingEmitter) EmitMessageEnd(text string, annotations []pipeline.Annotation) error {
	return nil
}

type recordingEmitter struct {
	searchCalls []struct {
		status string
		reason string
	}
}

func (e *recordingEmitter) EmitSearchCall(id, status, query, reason string, created int64, model string) error {
	e.searchCalls = append(e.searchCalls, struct {
		status string
		reason string
	}{status: status, reason: reason})
	return nil
}
func (e *recordingEmitter) EmitFetchCall(id, status, url string, created int64, model string) error {
	return nil
}
func (e *recordingEmitter) EmitMetadata(id string, created int64, model string, annotations []pipeline.Annotation) error {
	return nil
}
func (e *recordingEmitter) EmitChunk(data []byte) error { return nil }
func (e *recordingEmitter) EmitError(err error) error   { return nil }
func (e *recordingEmitter) EmitDone(id string, created int64, model string, usage openai.CompletionUsage) error {
	return nil
}
func (e *recordingEmitter) EmitResponseStart() error             { return nil }
func (e *recordingEmitter) EmitMessageStart(itemID string) error { return nil }
func (e *recordingEmitter) EmitMessageEnd(text string, annotations []pipeline.Annotation) error {
	return nil
}

type signalingEmitter struct {
	contentEmitted chan struct{}
}

func (e *signalingEmitter) EmitSearchCall(id, status, query, reason string, created int64, model string) error {
	return nil
}
func (e *signalingEmitter) EmitFetchCall(id, status, url string, created int64, model string) error {
	return nil
}
func (e *signalingEmitter) EmitMetadata(id string, created int64, model string, annotations []pipeline.Annotation) error {
	return nil
}
func (e *signalingEmitter) EmitChunk(data []byte) error {
	var chunk struct {
		Choices []struct {
			Delta struct {
				Content string `json:"content"`
			} `json:"delta"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(data, &chunk); err != nil {
		return nil
	}
	if len(chunk.Choices) == 0 || chunk.Choices[0].Delta.Content == "" {
		return nil
	}

	select {
	case e.contentEmitted <- struct{}{}:
	default:
	}
	return nil
}
func (e *signalingEmitter) EmitError(err error) error { return nil }
func (e *signalingEmitter) EmitDone(id string, created int64, model string, usage openai.CompletionUsage) error {
	return nil
}
func (e *signalingEmitter) EmitResponseStart() error             { return nil }
func (e *signalingEmitter) EmitMessageStart(itemID string) error { return nil }
func (e *signalingEmitter) EmitMessageEnd(text string, annotations []pipeline.Annotation) error {
	return nil
}

type waitingStream struct {
	events          []responses.ResponseStreamEventUnion
	index           int
	err             error
	contentEmitted  <-chan struct{}
	waitForEmitType string
}

func (s *waitingStream) Next() bool {
	if s.index > 0 && s.events[s.index-1].Type == s.waitForEmitType {
		select {
		case <-s.contentEmitted:
		case <-time.After(100 * time.Millisecond):
			s.err = fmt.Errorf("content was not emitted before advancing the stream")
			return false
		}
	}
	if s.index >= len(s.events) {
		return false
	}
	s.index++
	return true
}

func (s *waitingStream) Current() responses.ResponseStreamEventUnion {
	return s.events[s.index-1]
}

func (s *waitingStream) Err() error { return s.err }

func TestRun_MultiStepUsesPreviousResponseIDAndTracksSources(t *testing.T) {
	client := &fakeResponsesClient{
		responses: []*responses.Response{
			mustResponse(t, `{"id":"resp_1","created_at":1,"model":"gpt-oss-120b","output":[{"id":"fc_1","type":"function_call","call_id":"call_search","name":"search","arguments":"{\"query\":\"golang\"}"}],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`),
			mustResponse(t, `{"id":"resp_2","created_at":2,"model":"gpt-oss-120b","output":[{"id":"fc_2","type":"function_call","call_id":"call_fetch","name":"fetch","arguments":"{\"url\":\"https://go.dev/doc\"}"}],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`),
			mustResponse(t, `{"id":"resp_3","created_at":3,"model":"gpt-oss-120b","output":[{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"Go info【1】 and docs【2】","annotations":[]}]}],"usage":{"input_tokens":4,"output_tokens":6,"total_tokens":10}}`),
		},
	}
	service := NewService(
		client,
		&fakeSearcher{results: map[string][]search.Result{
			"golang": {{Title: "The Go Programming Language", URL: "https://go.dev", Content: "Go is an open source programming language."}},
		}},
		&fakeFetcher{pages: map[string]fetch.FetchedPage{
			"https://go.dev/doc": {URL: "https://go.dev/doc", Content: "Official Go documentation."},
		}},
		nil,
	)

	result, err := service.Run(context.Background(), chatRequest())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(client.params) != 3 {
		t.Fatalf("expected 3 model calls, got %d", len(client.params))
	}
	assertContainsJSON(t, client.params[1], `"call_id":"call_search"`)
	assertContainsJSON(t, client.params[1], `"name":"search"`)
	assertContainsJSON(t, client.params[2], `"call_id":"call_fetch"`)
	assertContainsJSON(t, client.params[2], `"name":"fetch"`)

	if len(result.Annotations) != 2 {
		t.Fatalf("expected 2 annotations, got %d", len(result.Annotations))
	}
	if result.Annotations[0].URLCitation.URL != "https://go.dev" {
		t.Fatalf("expected first annotation URL to be search result, got %s", result.Annotations[0].URLCitation.URL)
	}
	if result.Annotations[1].URLCitation.URL != "https://go.dev/doc" {
		t.Fatalf("expected second annotation URL to be fetched page, got %s", result.Annotations[1].URLCitation.URL)
	}
}

func TestRun_MixedTextAndFunctionCallContinuesCleanly(t *testing.T) {
	client := &fakeResponsesClient{
		responses: []*responses.Response{
			mustResponse(t, `{"id":"resp_mix_1","created_at":1,"model":"gpt-oss-120b","output":[{"id":"msg_pre","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"Let me check that.","annotations":[]}]},{"id":"fc_1","type":"function_call","call_id":"call_search","name":"search","arguments":"{\"query\":\"golang\"}"}],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`),
			mustResponse(t, `{"id":"resp_mix_2","created_at":2,"model":"gpt-oss-120b","output":[{"id":"msg_final","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"Here is the answer【1】","annotations":[]}]}],"usage":{"input_tokens":2,"output_tokens":2,"total_tokens":4}}`),
		},
	}
	service := NewService(
		client,
		&fakeSearcher{results: map[string][]search.Result{
			"golang": {{Title: "Go", URL: "https://go.dev", Content: "Go info"}},
		}},
		nil,
		nil,
	)

	result, err := service.Run(context.Background(), chatRequest())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	assertContainsJSON(t, client.params[1], `Let me check that.`)
	assertContainsJSON(t, client.params[1], `"call_id":"call_search"`)
	assertContainsJSON(t, client.params[1], `"name":"search"`)
	if result.Content != "Let me check that.Here is the answer【1】" {
		t.Fatalf("unexpected final content: %q", result.Content)
	}
}

func TestRun_BlockedSearchContinuesWithToolOutput(t *testing.T) {
	client := &fakeResponsesClient{
		responses: []*responses.Response{
			mustResponse(t, `{"id":"resp_block_1","created_at":1,"model":"gpt-oss-120b","output":[{"id":"fc_1","type":"function_call","call_id":"call_search","name":"search","arguments":"{\"query\":\"john@example.com\"}"}],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`),
			mustResponse(t, `{"id":"resp_block_2","created_at":2,"model":"gpt-oss-120b","output":[{"id":"msg_final","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"I can answer without searching.","annotations":[]}]}],"usage":{"input_tokens":2,"output_tokens":2,"total_tokens":4}}`),
		},
	}
	service := NewService(client, &fakeSearcher{}, nil, &fakeSafeguard{
		blocked: map[string]string{"john@example.com": "email detected"},
	})

	result, err := service.Run(context.Background(), chatRequestWithPII())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(result.BlockedQueries) != 1 {
		t.Fatalf("expected 1 blocked query, got %d", len(result.BlockedQueries))
	}
	assertContainsJSON(t, client.params[1], `Search blocked: email detected`)
}

func TestStream_UsesPreviousResponseIDAndEmitsMatchingAnnotations(t *testing.T) {
	client := &fakeResponsesClient{
		streams: []ResponseStream{
			&fakeStream{events: []responses.ResponseStreamEventUnion{
				mustEvent(t, `{"type":"response.created","sequence_number":1,"response":{"id":"resp_stream_1","created_at":1,"model":"gpt-oss-120b","object":"response","output":[],"parallel_tool_calls":false,"temperature":0}}`),
				mustEvent(t, `{"type":"response.output_item.added","sequence_number":2,"output_index":0,"item":{"id":"fc_1","type":"function_call","call_id":"call_search","name":"search","arguments":""}}`),
				mustEvent(t, `{"type":"response.function_call_arguments.done","sequence_number":3,"output_index":0,"item_id":"fc_1","name":"search","arguments":"{\"query\":\"golang\"}"}`),
			}},
			&fakeStream{events: []responses.ResponseStreamEventUnion{
				mustEvent(t, `{"type":"response.created","sequence_number":1,"response":{"id":"resp_stream_2","created_at":2,"model":"gpt-oss-120b","object":"response","output":[],"parallel_tool_calls":false,"temperature":0}}`),
				mustEvent(t, `{"type":"response.output_item.added","sequence_number":2,"output_index":0,"item":{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[]}}`),
				mustEvent(t, `{"type":"response.output_text.delta","sequence_number":3,"output_index":0,"content_index":0,"item_id":"msg_1","delta":"Go answer","logprobs":[]}`),
				mustEvent(t, `{"type":"response.output_text.delta","sequence_number":4,"output_index":0,"content_index":0,"item_id":"msg_1","delta":"【1】","logprobs":[]}`),
			}},
		},
	}
	service := NewService(
		client,
		&fakeSearcher{results: map[string][]search.Result{
			"golang": {{Title: "Go", URL: "https://go.dev", Content: "Go info"}},
		}},
		nil,
		nil,
	)
	emitter := &captureEmitter{}

	result, err := service.Stream(context.Background(), chatRequest(), emitter)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	assertContainsJSON(t, client.params[1], `"call_id":"call_search"`)
	assertContainsJSON(t, client.params[1], `"name":"search"`)
	if len(emitter.annotations) != 1 || len(result.Annotations) != 1 {
		t.Fatalf("expected one annotation from streaming result")
	}
	if emitter.annotations[0].URLCitation.URL != result.Annotations[0].URLCitation.URL {
		t.Fatalf("expected emitter and result annotation URLs to match")
	}
	if !strings.Contains(strings.Join(emitter.chunks, ""), `Go answer`) {
		t.Fatalf("expected emitted chunks to contain final answer")
	}
}

func TestStream_MixedTextAndFunctionCallContinuesCleanly(t *testing.T) {
	client := &fakeResponsesClient{
		streams: []ResponseStream{
			&fakeStream{events: []responses.ResponseStreamEventUnion{
				mustEvent(t, `{"type":"response.created","sequence_number":1,"response":{"id":"resp_stream_mix_1","created_at":1,"model":"gpt-oss-120b","object":"response","output":[],"parallel_tool_calls":false,"temperature":0}}`),
				mustEvent(t, `{"type":"response.output_item.added","sequence_number":2,"output_index":0,"item":{"id":"msg_pre","type":"message","role":"assistant","status":"completed","content":[]}}`),
				mustEvent(t, `{"type":"response.output_text.delta","sequence_number":3,"output_index":0,"content_index":0,"item_id":"msg_pre","delta":"Let me check that.","logprobs":[]}`),
				mustEvent(t, `{"type":"response.output_item.added","sequence_number":4,"output_index":1,"item":{"id":"fc_1","type":"function_call","call_id":"call_search","name":"search","arguments":""}}`),
				mustEvent(t, `{"type":"response.function_call_arguments.done","sequence_number":5,"output_index":1,"item_id":"fc_1","name":"search","arguments":"{\"query\":\"golang\"}"}`),
			}},
			&fakeStream{events: []responses.ResponseStreamEventUnion{
				mustEvent(t, `{"type":"response.created","sequence_number":1,"response":{"id":"resp_stream_mix_2","created_at":2,"model":"gpt-oss-120b","object":"response","output":[],"parallel_tool_calls":false,"temperature":0}}`),
				mustEvent(t, `{"type":"response.output_item.added","sequence_number":2,"output_index":0,"item":{"id":"msg_final","type":"message","role":"assistant","status":"completed","content":[]}}`),
				mustEvent(t, `{"type":"response.output_text.delta","sequence_number":3,"output_index":0,"content_index":0,"item_id":"msg_final","delta":"Here is the answer【1】","logprobs":[]}`),
			}},
		},
	}
	service := NewService(
		client,
		&fakeSearcher{results: map[string][]search.Result{
			"golang": {{Title: "Go", URL: "https://go.dev", Content: "Go info"}},
		}},
		nil,
		nil,
	)
	emitter := &captureEmitter{}

	result, err := service.Stream(context.Background(), chatRequest(), emitter)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	assertContainsJSON(t, client.params[1], `Let me check that.`)
	assertContainsJSON(t, client.params[1], `"call_id":"call_search"`)
	assertContainsJSON(t, client.params[1], `"name":"search"`)
	if result.Content != "Let me check that.Here is the answer【1】" {
		t.Fatalf("unexpected final content: %q", result.Content)
	}
	if !strings.Contains(strings.Join(emitter.chunks, ""), `Let me check that.`) {
		t.Fatalf("expected emitted chunks to contain interim text")
	}
	if !strings.Contains(strings.Join(emitter.chunks, ""), `Here is the answer`) {
		t.Fatalf("expected emitted chunks to contain final text")
	}
}

func TestStream_EmitsContentBeforeAdvancingFinalAnswerStream(t *testing.T) {
	contentEmitted := make(chan struct{}, 1)
	client := &fakeResponsesClient{
		streams: []ResponseStream{
			&fakeStream{events: []responses.ResponseStreamEventUnion{
				mustEvent(t, `{"type":"response.created","sequence_number":1,"response":{"id":"resp_stream_1","created_at":1,"model":"gpt-oss-120b","object":"response","output":[],"parallel_tool_calls":false,"temperature":0}}`),
				mustEvent(t, `{"type":"response.output_item.added","sequence_number":2,"output_index":0,"item":{"id":"fc_1","type":"function_call","call_id":"call_search","name":"search","arguments":""}}`),
				mustEvent(t, `{"type":"response.function_call_arguments.done","sequence_number":3,"output_index":0,"item_id":"fc_1","name":"search","arguments":"{\"query\":\"golang\"}"}`),
			}},
			&waitingStream{
				waitForEmitType: "response.output_text.delta",
				contentEmitted:  contentEmitted,
				events: []responses.ResponseStreamEventUnion{
					mustEvent(t, `{"type":"response.created","sequence_number":1,"response":{"id":"resp_stream_2","created_at":2,"model":"gpt-oss-120b","object":"response","output":[],"parallel_tool_calls":false,"temperature":0}}`),
					mustEvent(t, `{"type":"response.output_item.added","sequence_number":2,"output_index":0,"item":{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[]}}`),
					mustEvent(t, `{"type":"response.output_text.delta","sequence_number":3,"output_index":0,"content_index":0,"item_id":"msg_1","delta":"Go answer","logprobs":[]}`),
					mustEvent(t, `{"type":"response.output_text.delta","sequence_number":4,"output_index":0,"content_index":0,"item_id":"msg_1","delta":"【1】","logprobs":[]}`),
				},
			},
		},
	}
	service := NewService(
		client,
		&fakeSearcher{results: map[string][]search.Result{
			"golang": {{Title: "Go", URL: "https://go.dev", Content: "Go info"}},
		}},
		nil,
		nil,
	)

	if _, err := service.Stream(context.Background(), chatRequest(), &signalingEmitter{contentEmitted: contentEmitted}); err != nil {
		t.Fatalf("expected content to stream before the upstream stream advanced: %v", err)
	}
}

func TestStream_AggregatesUsageAcrossStreamingResponses(t *testing.T) {
	client := &fakeResponsesClient{
		streams: []ResponseStream{
			&fakeStream{events: []responses.ResponseStreamEventUnion{
				mustEvent(t, `{"type":"response.created","sequence_number":1,"response":{"id":"resp_stream_1","created_at":1,"model":"gpt-oss-120b","object":"response","output":[],"parallel_tool_calls":false,"temperature":0}}`),
				mustEvent(t, `{"type":"response.output_item.added","sequence_number":2,"output_index":0,"item":{"id":"fc_1","type":"function_call","call_id":"call_search","name":"search","arguments":""}}`),
				mustEvent(t, `{"type":"response.function_call_arguments.done","sequence_number":3,"output_index":0,"item_id":"fc_1","name":"search","arguments":"{\"query\":\"golang\"}"}`),
				mustEvent(t, `{"type":"response.completed","sequence_number":4,"response":{"id":"resp_stream_1","created_at":1,"model":"gpt-oss-120b","object":"response","output":[],"usage":{"input_tokens":1,"output_tokens":2,"total_tokens":3}}}`),
			}},
			&fakeStream{events: []responses.ResponseStreamEventUnion{
				mustEvent(t, `{"type":"response.created","sequence_number":1,"response":{"id":"resp_stream_2","created_at":2,"model":"gpt-oss-120b","object":"response","output":[],"parallel_tool_calls":false,"temperature":0}}`),
				mustEvent(t, `{"type":"response.output_item.added","sequence_number":2,"output_index":0,"item":{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[]}}`),
				mustEvent(t, `{"type":"response.output_text.delta","sequence_number":3,"output_index":0,"content_index":0,"item_id":"msg_1","delta":"Go answer","logprobs":[]}`),
				mustEvent(t, `{"type":"response.completed","sequence_number":4,"response":{"id":"resp_stream_2","created_at":2,"model":"gpt-oss-120b","object":"response","output":[],"usage":{"input_tokens":4,"output_tokens":5,"total_tokens":9}}}`),
			}},
		},
	}
	service := NewService(
		client,
		&fakeSearcher{results: map[string][]search.Result{
			"golang": {{Title: "Go", URL: "https://go.dev", Content: "Go info"}},
		}},
		nil,
		nil,
	)

	result, err := service.Stream(context.Background(), chatRequest(), &captureEmitter{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.Usage.PromptTokens != 5 || result.Usage.CompletionTokens != 7 || result.Usage.TotalTokens != 12 {
		t.Fatalf("unexpected aggregated usage: %+v", result.Usage)
	}
}

func TestRun_CompactsFetchedToolOutputWithSummaryModel(t *testing.T) {
	longPage := strings.Repeat("Long fetched content. ", 1500)
	client := &fakeResponsesClient{
		responses: []*responses.Response{
			mustResponse(t, `{"id":"resp_fetch_1","created_at":1,"model":"gpt-oss-120b","output":[{"id":"fc_1","type":"function_call","call_id":"call_fetch","name":"fetch","arguments":"{\"url\":\"https://go.dev/doc\"}"}],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`),
			mustResponse(t, `{"id":"resp_summary_1","created_at":2,"model":"gpt-oss-120b","output":[{"id":"msg_summary","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"【1】Fetched page\nURL: https://go.dev/doc\nKey facts only.","annotations":[]}]}],"usage":{"input_tokens":2,"output_tokens":2,"total_tokens":4}}`),
			mustResponse(t, `{"id":"resp_fetch_2","created_at":3,"model":"gpt-oss-120b","output":[{"id":"msg_final","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"Docs say key facts【1】","annotations":[]}]}],"usage":{"input_tokens":2,"output_tokens":2,"total_tokens":4}}`),
		},
	}
	service := NewService(
		client,
		nil,
		&fakeFetcher{pages: map[string]fetch.FetchedPage{
			"https://go.dev/doc": {URL: "https://go.dev/doc", Content: longPage},
		}},
		nil,
		WithToolSummaryModel("gpt-oss-120b"),
	)

	result, err := service.Run(context.Background(), chatRequest())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(client.params) != 3 {
		t.Fatalf("expected 3 model calls, got %d", len(client.params))
	}
	assertContainsJSON(t, client.params[1], "You compress web tool output")
	assertContainsJSON(t, client.params[2], "Key facts only.")
	assertNotContainsJSON(t, client.params[2], longPage)
	if result.Content != "Docs say key facts【1】" {
		t.Fatalf("unexpected final content: %q", result.Content)
	}
}

func TestSearch_AllowsSearchWhenPIICheckErrors(t *testing.T) {
	service := NewService(nil, &fakeSearcher{results: map[string][]search.Result{
		"john@example.com": {{Title: "Example", URL: "https://example.com", Content: "allowed"}},
	}}, nil, &fakeSafeguard{
		errs: map[string]error{"john@example.com": fmt.Errorf("safeguard unavailable")},
	})

	outcome, err := service.Search(context.Background(), "john@example.com", ToolOptions{PIICheckEnabled: true})
	if err != nil {
		t.Fatalf("expected search to remain fail-open, got %v", err)
	}
	if len(outcome.Results) != 1 {
		t.Fatalf("expected search results to be preserved, got %d", len(outcome.Results))
	}
}

func TestSearch_FiltersResultsWhenInjectionCheckEnabled(t *testing.T) {
	service := NewService(nil, &fakeSearcher{results: map[string][]search.Result{
		"golang": {
			{Title: "Safe", URL: "https://example.com/safe", Content: "safe"},
			{Title: "Unsafe", URL: "https://example.com/unsafe", Content: "unsafe"},
		},
	}}, nil, &fakeSafeguard{
		blocked: map[string]string{"unsafe": "prompt injection detected"},
	})

	outcome, err := service.Search(context.Background(), "golang", ToolOptions{InjectionCheckEnabled: true})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(outcome.Results) != 1 {
		t.Fatalf("expected one filtered search result, got %d", len(outcome.Results))
	}
	if outcome.Results[0].Title != "Safe" {
		t.Fatalf("expected safe result to remain, got %q", outcome.Results[0].Title)
	}
}

func TestFilterSearchResults_KeepsResultsWhenInjectionCheckErrors(t *testing.T) {
	results := filterSearchResults(context.Background(), &fakeSafeguard{
		errs: map[string]error{"unsafe": fmt.Errorf("timeout")},
	}, []search.Result{{Title: "Unsafe", URL: "https://example.com", Content: "unsafe"}})

	if len(results) != 1 {
		t.Fatalf("expected safeguard errors to keep the result, got %d results", len(results))
	}
}

func TestFilterFetchedPages_KeepsPagesWhenInjectionCheckErrors(t *testing.T) {
	pages := filterFetchedPages(context.Background(), &fakeSafeguard{
		errs: map[string]error{"unsafe": fmt.Errorf("timeout")},
	}, []fetch.FetchedPage{{URL: "https://example.com", Content: "unsafe"}})

	if len(pages) != 1 {
		t.Fatalf("expected safeguard errors to keep the page, got %d pages", len(pages))
	}
}

func TestFilterFetchResults_KeepsResultsWhenInjectionCheckErrors(t *testing.T) {
	results := filterFetchResults(context.Background(), &fakeSafeguard{
		errs: map[string]error{"unsafe": fmt.Errorf("timeout")},
	}, []fetch.URLResult{{
		URL:     "https://example.com",
		Status:  fetch.FetchStatusCompleted,
		Content: "unsafe",
	}})

	if len(results) != 1 {
		t.Fatalf("expected one fetch result, got %d", len(results))
	}
	if results[0].Status != fetch.FetchStatusCompleted {
		t.Fatalf("expected fetch status to remain completed, got %q", results[0].Status)
	}
	if results[0].Content != "unsafe" {
		t.Fatalf("expected fetch content to be preserved, got %q", results[0].Content)
	}
	if results[0].Error != "" {
		t.Fatalf("expected no fetch error, got %q", results[0].Error)
	}
}

func TestExecuteToolCalls_ReportsSearchFailureReason(t *testing.T) {
	service := NewService(nil, &fakeSearcher{err: fmt.Errorf("search backend unavailable")}, nil, nil)
	emitter := &recordingEmitter{}
	req := chatRequest()
	calls := map[int]*functionCall{
		0: {index: 0, id: "call_search", name: "search", arguments: strings.Builder{}},
	}
	calls[0].arguments.WriteString(`{"query":"golang"}`)

	_, _ = service.executeToolCalls(context.Background(), req, calls, emitter)

	if len(emitter.searchCalls) < 2 {
		t.Fatalf("expected in_progress and failed events, got %d", len(emitter.searchCalls))
	}
	lastCall := emitter.searchCalls[len(emitter.searchCalls)-1]
	if lastCall.status != "failed" {
		t.Fatalf("expected final search status failed, got %q", lastCall.status)
	}
	if lastCall.reason != "search backend unavailable" {
		t.Fatalf("expected failure reason to be propagated, got %q", lastCall.reason)
	}
}

func TestBuildAnnotationsFromContent_UsesRuneIndexes(t *testing.T) {
	annotations := buildAnnotationsFromContent("éclair【1】", []CitationSource{{
		Title: "Example",
		URL:   "https://example.com",
	}})

	if len(annotations) != 1 {
		t.Fatalf("expected one annotation, got %d", len(annotations))
	}
	if annotations[0].URLCitation.StartIndex != 6 {
		t.Fatalf("expected rune start index 6, got %d", annotations[0].URLCitation.StartIndex)
	}
	if annotations[0].URLCitation.EndIndex != 9 {
		t.Fatalf("expected rune end index 9, got %d", annotations[0].URLCitation.EndIndex)
	}
}

func TestExecuteToolCalls_SerializesEmitterWrites(t *testing.T) {
	service := NewService(
		nil,
		&fakeSearcher{results: map[string][]search.Result{
			"golang": {{Title: "Go", URL: "https://go.dev", Content: "Go info"}},
		}},
		&fakeFetcher{pages: map[string]fetch.FetchedPage{
			"https://go.dev/doc": {URL: "https://go.dev/doc", Content: "Official docs"},
		}},
		nil,
	)
	emitter := &serializingEmitter{}
	req := chatRequest()

	calls := map[int]*functionCall{
		0: {index: 0, id: "call_search", name: "search", arguments: strings.Builder{}},
		1: {index: 1, id: "call_fetch", name: "fetch", arguments: strings.Builder{}},
	}
	calls[0].arguments.WriteString(`{"query":"golang"}`)
	calls[1].arguments.WriteString(`{"url":"https://go.dev/doc"}`)

	_, _ = service.executeToolCalls(context.Background(), req, calls, emitter)

	if emitter.maxConcurrent.Load() > 1 {
		t.Fatalf("expected emitter writes to be serialized, saw concurrency=%d", emitter.maxConcurrent.Load())
	}
}

func TestRun_ResponsesRequestSupportsStructuredInputAndClientPreviousResponseID(t *testing.T) {
	client := &fakeResponsesClient{
		responses: []*responses.Response{
			mustResponse(t, `{"id":"resp_final","created_at":3,"model":"gpt-oss-120b","output":[{"id":"msg_final","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"Done","annotations":[]}]}],"usage":{"input_tokens":2,"output_tokens":2,"total_tokens":4}}`),
		},
	}
	service := NewService(client, nil, nil, nil)

	result, err := service.Run(context.Background(), &pipeline.Request{
		Model:              "gpt-oss-120b",
		Format:             pipeline.FormatResponses,
		Input:              json.RawMessage(`[{"role":"developer","content":"Be concise."},{"type":"message","role":"user","content":[{"type":"input_text","text":"hello"}]}]`),
		PreviousResponseID: "resp_prev_client",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.ID != "resp_final" {
		t.Fatalf("expected final response id, got %q", result.ID)
	}
	if len(client.params) != 1 {
		t.Fatalf("expected 1 model call, got %d", len(client.params))
	}
	assertContainsJSON(t, client.params[0], `"previous_response_id":"resp_prev_client"`)
	assertContainsJSON(t, client.params[0], `"role":"developer"`)
	assertContainsJSON(t, client.params[0], `"text":"hello"`)
}

func TestRun_ResponsesRequestSupportsReasoningAndFunctionCallInputItems(t *testing.T) {
	client := &fakeResponsesClient{
		responses: []*responses.Response{
			mustResponse(t, `{"id":"resp_final","created_at":3,"model":"gpt-oss-120b","output":[{"id":"msg_final","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"Done","annotations":[]}]}],"usage":{"input_tokens":2,"output_tokens":2,"total_tokens":4}}`),
		},
	}
	service := NewService(client, nil, nil, nil)

	input := `[
		{"role":"user","content":"Search for cats"},
		{"type":"reasoning","id":"rs_abc","content":[{"type":"reasoning_text","text":"thinking about cats"}],"summary":[]},
		{"type":"function_call","call_id":"call_1","name":"search","arguments":"{\"query\":\"cats\"}"},
		{"type":"function_call_output","call_id":"call_1","output":"cats are great"},
		{"role":"assistant","content":"Here is info about cats."},
		{"role":"user","content":"Tell me more"}
	]`

	result, err := service.Run(context.Background(), &pipeline.Request{
		Model:  "gpt-oss-120b",
		Format: pipeline.FormatResponses,
		Input:  json.RawMessage(input),
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Content != "Done" {
		t.Fatalf("expected 'Done', got %q", result.Content)
	}
	if len(client.params) != 1 {
		t.Fatalf("expected 1 model call, got %d", len(client.params))
	}
	assertContainsJSON(t, client.params[0], `"thinking about cats"`)
	assertContainsJSON(t, client.params[0], `"call_id":"call_1"`)
	assertContainsJSON(t, client.params[0], `"name":"search"`)
	assertContainsJSON(t, client.params[0], `"cats are great"`)
}

func chatRequest() *pipeline.Request {
	return &pipeline.Request{
		Model:            "gpt-oss-120b",
		Format:           pipeline.FormatChatCompletion,
		WebSearchEnabled: true,
		Messages: []pipeline.Message{
			{Role: "user", Content: mustJSONRaw("Tell me about Go")},
		},
	}
}

func chatRequestWithPII() *pipeline.Request {
	req := chatRequest()
	req.PIICheckEnabled = true
	return req
}

func mustJSONRaw(text string) json.RawMessage {
	data, _ := json.Marshal(text)
	return data
}

func mustResponse(t *testing.T, raw string) *responses.Response {
	t.Helper()
	var resp responses.Response
	if err := json.Unmarshal([]byte(raw), &resp); err != nil {
		t.Fatalf("failed to unmarshal response: %v", err)
	}
	return &resp
}

func mustEvent(t *testing.T, raw string) responses.ResponseStreamEventUnion {
	t.Helper()
	var event responses.ResponseStreamEventUnion
	if err := json.Unmarshal([]byte(raw), &event); err != nil {
		t.Fatalf("failed to unmarshal event: %v", err)
	}
	return event
}

func assertContainsJSON(t *testing.T, params responses.ResponseNewParams, needle string) {
	t.Helper()
	data, err := json.Marshal(params)
	if err != nil {
		t.Fatalf("failed to marshal params: %v", err)
	}
	if !strings.Contains(string(data), needle) {
		t.Fatalf("expected marshaled params to contain %q, got %s", needle, string(data))
	}
}

func assertNotContainsJSON(t *testing.T, params responses.ResponseNewParams, needle string) {
	t.Helper()
	data, err := json.Marshal(params)
	if err != nil {
		t.Fatalf("failed to marshal params: %v", err)
	}
	if strings.Contains(string(data), needle) {
		t.Fatalf("expected marshaled params not to contain %q, got %s", needle, string(data))
	}
}
