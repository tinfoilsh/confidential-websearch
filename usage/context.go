package usage

import (
	"net/http"
	"strings"
)

const (
	headerRequestID = "X-Tinfoil-Tool-Request-Id"
	headerModel     = "X-Tinfoil-Tool-Model"
	headerRoute     = "X-Tinfoil-Tool-Route"
	headerStreaming = "X-Tinfoil-Tool-Streaming"
)

type requestContext struct {
	RequestID  string
	Model      string
	Route      string
	Streaming  bool
	AuthHeader string
}

func contextFromRequest(r *http.Request) requestContext {
	if r == nil {
		return requestContext{}
	}
	return requestContext{
		RequestID:  strings.TrimSpace(r.Header.Get(headerRequestID)),
		Model:      strings.TrimSpace(r.Header.Get(headerModel)),
		Route:      strings.TrimSpace(r.Header.Get(headerRoute)),
		Streaming:  strings.EqualFold(strings.TrimSpace(r.Header.Get(headerStreaming)), "true"),
		AuthHeader: strings.TrimSpace(r.Header.Get("Authorization")),
	}
}

func bearerToken(authHeader string) string {
	parts := strings.Fields(strings.TrimSpace(authHeader))
	if len(parts) == 2 && strings.EqualFold(parts[0], "Bearer") {
		return parts[1]
	}
	return ""
}
