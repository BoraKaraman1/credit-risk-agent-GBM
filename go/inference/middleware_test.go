package inference

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/prometheus/client_golang/prometheus/testutil"
)

func okHandler() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok"))
	})
}

func TestKeyAllowed(t *testing.T) {
	keys := [][]byte{[]byte("alpha"), []byte("bravo")}
	cases := []struct {
		key  string
		want bool
	}{
		{"alpha", true},
		{"bravo", true},
		{"charlie", false},
		{"", false},
		{"alph", false},
	}
	for _, tc := range cases {
		if got := keyAllowed(keys, tc.key); got != tc.want {
			t.Errorf("keyAllowed(%q) = %v, want %v", tc.key, got, tc.want)
		}
	}
	if keyAllowed(nil, "alpha") {
		t.Error("empty key set should reject everything")
	}
}

func TestAuthMiddleware(t *testing.T) {
	withKeys := &server{apiKeys: parseAPIKeys([]string{"secret"})}

	t.Run("disabled passes through when no keys", func(t *testing.T) {
		s := &server{}
		rec := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodPost, "/score", nil)
		s.authMiddleware(okHandler()).ServeHTTP(rec, req)
		if rec.Code != http.StatusOK {
			t.Errorf("status = %d, want 200", rec.Code)
		}
	})
	t.Run("401 without key", func(t *testing.T) {
		rec := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodPost, "/score", nil)
		withKeys.authMiddleware(okHandler()).ServeHTTP(rec, req)
		if rec.Code != http.StatusUnauthorized {
			t.Errorf("status = %d, want 401", rec.Code)
		}
	})
	t.Run("401 with wrong key", func(t *testing.T) {
		rec := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodPost, "/score", nil)
		req.Header.Set("X-API-Key", "nope")
		withKeys.authMiddleware(okHandler()).ServeHTTP(rec, req)
		if rec.Code != http.StatusUnauthorized {
			t.Errorf("status = %d, want 401", rec.Code)
		}
	})
	t.Run("200 with valid key, key stashed for rate limiting", func(t *testing.T) {
		rec := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodPost, "/score", nil)
		req.Header.Set("X-API-Key", "secret")
		var gotClient string
		next := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			gotClient = clientID(r)
			w.WriteHeader(http.StatusOK)
		})
		withKeys.authMiddleware(next).ServeHTTP(rec, req)
		if rec.Code != http.StatusOK {
			t.Fatalf("status = %d, want 200", rec.Code)
		}
		if gotClient != "key:secret" {
			t.Errorf("clientID = %q, want key:secret", gotClient)
		}
	})
}

func TestRateLimitMiddleware(t *testing.T) {
	s := &server{limiter: newRateLimiter(1, 3)} // 1 rps, burst 3
	h := s.rateLimitMiddleware(okHandler())

	send := func(ip string) int {
		rec := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodPost, "/score", nil)
		req.RemoteAddr = ip + ":5555"
		h.ServeHTTP(rec, req)
		return rec.Code
	}

	for i := 0; i < 3; i++ {
		if code := send("1.2.3.4"); code != http.StatusOK {
			t.Fatalf("burst request %d: status %d, want 200", i, code)
		}
	}
	if code := send("1.2.3.4"); code != http.StatusTooManyRequests {
		t.Errorf("over-budget request: status %d, want 429", code)
	}
	if code := send("9.9.9.9"); code != http.StatusOK {
		t.Errorf("different client: status %d, want 200 (own bucket)", code)
	}
}

func TestClientIP(t *testing.T) {
	cases := []struct {
		name       string
		remoteAddr string
		xff        string
		want       string
	}{
		{"remote addr with port", "1.2.3.4:5555", "", "1.2.3.4"},
		{"x-forwarded-for single", "10.0.0.1:1", "203.0.113.5", "203.0.113.5"},
		{"x-forwarded-for chain takes first", "10.0.0.1:1", "203.0.113.1, 70.41.3.18", "203.0.113.1"},
		{"remote addr without port", "unixsocket", "", "unixsocket"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, "/score", nil)
			req.RemoteAddr = tc.remoteAddr
			if tc.xff != "" {
				req.Header.Set("X-Forwarded-For", tc.xff)
			}
			if got := clientIP(req); got != tc.want {
				t.Errorf("clientIP = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestClientIDFallsBackToIP(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/score", nil)
	req.RemoteAddr = "8.8.8.8:1234"
	if got := clientID(req); got != "ip:8.8.8.8" {
		t.Errorf("clientID = %q, want ip:8.8.8.8", got)
	}
}

func TestInstrument(t *testing.T) {
	t.Run("generates a request id", func(t *testing.T) {
		rec := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodGet, "/health", nil)
		instrument("/health", okHandler()).ServeHTTP(rec, req)
		if rec.Header().Get("X-Request-ID") == "" {
			t.Error("response missing X-Request-ID")
		}
	})
	t.Run("propagates an incoming request id", func(t *testing.T) {
		rec := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodGet, "/health", nil)
		req.Header.Set("X-Request-ID", "trace-123")
		instrument("/health", okHandler()).ServeHTTP(rec, req)
		if got := rec.Header().Get("X-Request-ID"); got != "trace-123" {
			t.Errorf("X-Request-ID = %q, want trace-123", got)
		}
	})
	t.Run("request-scoped logger is available", func(t *testing.T) {
		var ok bool
		h := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			ok = logger(r.Context()) != nil
		})
		rec := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodGet, "/health", nil)
		instrument("/health", h).ServeHTTP(rec, req)
		if !ok {
			t.Error("logger(ctx) not set inside instrumented handler")
		}
	})
	t.Run("records status in the request counter", func(t *testing.T) {
		const endpoint = "/test-instrument"
		h := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusServiceUnavailable)
		})
		before := testutil.ToFloat64(httpRequests.WithLabelValues(endpoint, http.MethodGet, "503"))
		rec := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodGet, endpoint, nil)
		instrument(endpoint, h).ServeHTTP(rec, req)
		after := testutil.ToFloat64(httpRequests.WithLabelValues(endpoint, http.MethodGet, "503"))
		if after-before != 1 {
			t.Errorf("counter delta = %v, want 1", after-before)
		}
	})
}

// logger() outside an instrumented request must not panic.
func TestLoggerFallback(t *testing.T) {
	if logger(context.Background()) == nil {
		t.Error("logger(background) returned nil")
	}
}
