package main

import (
	"errors"
	stdlog "log"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"strings"
	"testing"

	log "github.com/sirupsen/logrus"
)

const (
	loggingTestActionEnv     = "WEBSEARCH_LOGGING_TEST_ACTION"
	loggingTestVerboseEnv    = "WEBSEARCH_LOGGING_TEST_VERBOSE"
	loggingTestEmit          = "emit"
	loggingTestStartup       = "startup"
	loggingTestFatalExitCode = 1
)

func TestLoggingOnlyInLocalTestMode(t *testing.T) {
	for _, mode := range []string{"", "0", "false", "true", "1 ", "1"} {
		for _, verbose := range []string{"", "1"} {
			t.Run("mode="+mode+"/verbose="+verbose, func(t *testing.T) {
				t.Setenv("LOCAL_TEST_MODE", mode)
				t.Setenv(loggingTestVerboseEnv, verbose)
				output := runLoggingProcess(t, loggingTestEmit)
				if mode != "1" {
					if output != "" {
						t.Fatalf("production emitted logs: %s", output)
					}
					return
				}
				for _, message := range []string{"application info", "application warning", "application error", "standard log", "dependency warning", "http: panic serving", "request panic", "fatal error"} {
					if !strings.Contains(output, message) {
						t.Errorf("local logs missing %q: %s", message, output)
					}
				}
				if got, want := strings.Contains(output, "application debug"), verbose == "1"; got != want {
					t.Errorf("debug output present = %v, want %v", got, want)
				}
			})
		}
	}
}

func TestProductionStartupFailureIsSilent(t *testing.T) {
	t.Setenv("LOCAL_TEST_MODE", "")
	t.Setenv("USAGE_REPORTER_SECRET", "")
	if output := runLoggingProcess(t, loggingTestStartup); output != "" {
		t.Fatalf("production startup emitted logs with -v: %s", output)
	}
}

func runLoggingProcess(t *testing.T, action string) string {
	t.Helper()
	t.Setenv(loggingTestActionEnv, action)
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(executable, "-test.run=^TestLoggingProcess$")
	output, err := cmd.CombinedOutput()
	var exitErr *exec.ExitError
	if !errors.As(err, &exitErr) || exitErr.ExitCode() != loggingTestFatalExitCode {
		t.Fatalf("expected fatal exit %d, got %v: %s", loggingTestFatalExitCode, err, output)
	}
	return string(output)
}

func TestLoggingProcess(t *testing.T) {
	switch os.Getenv(loggingTestActionEnv) {
	case loggingTestStartup:
		os.Args = []string{"websearch-mcp", "-v"}
		main()
		t.Fatal("startup should fail without the usage reporter secret")
	case loggingTestEmit:
		configureLogging(isLocalTestMode())
		if os.Getenv(loggingTestVerboseEnv) == "1" {
			log.SetLevel(log.DebugLevel)
		}
		log.Debug("application debug")
		log.Info("application info")
		log.Warn("application warning")
		log.Error("application error")
		stdlog.Print("standard log")
		slog.Warn("dependency warning")

		server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
			panic("request panic")
		}))
		response, err := server.Client().Get(server.URL)
		if response != nil {
			response.Body.Close()
		}
		server.Close()
		if err == nil {
			t.Fatal("expected the panicking handler to close the connection")
		}
		log.Fatal("fatal error")
	}
}
