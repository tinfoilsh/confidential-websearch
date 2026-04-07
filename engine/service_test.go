package engine

import (
	"context"
	"encoding/json"
	"strings"
	"sync/atomic"
	"testing"
	"time"

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
}

func (s *fakeSearcher) Search(ctx context.Context, query string, opts search.Options) ([]search.Result, error) {
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
}

func (f *fakeSafeguard) Check(ctx context.Context, policy, content string) (*safeguard.CheckResult, error) {
	if reason, ok := f.blocked[content]; ok {
		return &safeguard.CheckResult{Violation: true, Rationale: reason}, nil
	}
	return &safeguard.CheckResult{}, nil
}

type captureEmitter struct {
	annotations []pipeline.Annotation
	reasoning   string
	chunks      []string
}

func (e *captureEmitter) EmitSearchCall(id, status, query, reason string, created int64, model string) error {
	return nil
}
func (e *captureEmitter) EmitFetchCall(id, status, url string, created int64, model string) error {
	return nil
}
func (e *captureEmitter) EmitMetadata(id string, created int64, model string, annotations []pipeline.Annotation, reasoning string) error {
	e.annotations = annotations
	e.reasoning = reasoning
	return nil
}
func (e *captureEmitter) EmitChunk(data []byte) error {
	e.chunks = append(e.chunks, string(data))
	return nil
}
func (e *captureEmitter) EmitError(err error) error { return nil }
func (e *captureEmitter) EmitDone() error           { return nil }
func (e *captureEmitter) EmitResponseStart() error  { return nil }
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

func (e *serializingEmitter) EmitMetadata(id string, created int64, model string, annotations []pipeline.Annotation, reasoning string) error {
	return nil
}
func (e *serializingEmitter) EmitChunk(data []byte) error          { return nil }
func (e *serializingEmitter) EmitError(err error) error            { return nil }
func (e *serializingEmitter) EmitDone() error                      { return nil }
func (e *serializingEmitter) EmitResponseStart() error             { return nil }
func (e *serializingEmitter) EmitMessageStart(itemID string) error { return nil }
func (e *serializingEmitter) EmitMessageEnd(text string, annotations []pipeline.Annotation) error {
	return nil
}

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
	assertContainsJSON(t, client.params[1], `"previous_response_id":"resp_1"`)
	assertContainsJSON(t, client.params[1], `"call_id":"call_search"`)
	assertContainsJSON(t, client.params[2], `"previous_response_id":"resp_2"`)
	assertContainsJSON(t, client.params[2], `"call_id":"call_fetch"`)

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

	assertContainsJSON(t, client.params[1], `"previous_response_id":"resp_mix_1"`)
	if result.Content != "Here is the answer【1】" {
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

	assertContainsJSON(t, client.params[1], `"previous_response_id":"resp_stream_1"`)
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

func TestRun_CompactsFetchedToolOutputWithSummaryModel(t *testing.T) {
	longPage := strings.Repeat("Long fetched content. ", 400)
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
