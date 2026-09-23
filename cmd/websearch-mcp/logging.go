package main

import (
	"io"
	stdlog "log"
	"log/slog"

	log "github.com/sirupsen/logrus"
)

func configureLogging(localTestMode bool) {
	if localTestMode {
		return
	}

	log.SetOutput(io.Discard)
	slog.SetDefault(slog.New(slog.DiscardHandler))
	stdlog.SetOutput(io.Discard)
}
