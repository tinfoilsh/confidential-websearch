FROM golang:1.27.4-alpine@sha256:70bd7ddea8b6cbe5d6943cde279d6b67feb10cc924fb9100d1d8757c1575b280 AS builder

WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY . .

RUN CGO_ENABLED=0 GOOS=linux go build -o websearch-mcp ./cmd/websearch-mcp

FROM alpine:3.23@sha256:5b10f432ef3da1b8d4c7eb6c487f2f5a8f096bc91145e68878dd4a5019afde11

# The apk version constraint fails the build when the pinned alpine
# image's repository offers a CA bundle older than the CVE-2026-111943 fix.
RUN apk --no-cache add 'ca-certificates>=20261020-r0'

WORKDIR /app

COPY --from=builder /app/websearch-mcp .

EXPOSE 8089

ENTRYPOINT ["./websearch-mcp"]
