package inference

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

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
			gotClient = clientID(r, false)
			w.WriteHeader(http.StatusOK)
		})
		withKeys.authMiddleware(next).ServeHTTP(rec, req)
		if rec.Code != http.StatusOK {
			t.Fatalf("status = %d, want 200", rec.Code)
		}
		// The limiter identity is a digest of the key — the raw
		// credential must never appear in limiter state or logs.
		sum := sha256.Sum256([]byte("secret"))
		want := "key:" + hex.EncodeToString(sum[:4])
		if gotClient != want {
			t.Errorf("clientID = %q, want %q", gotClient, want)
		}
		if strings.Contains(gotClient, "secret") {
			t.Errorf("clientID %q leaks the raw API key", gotClient)
		}
	})
}

func TestPresentedKey(t *testing.T) {
	cases := []struct {
		name   string
		header [2]string
		want   string
	}{
		{"x-api-key", [2]string{"X-API-Key", "abc"}, "abc"},
		{"bearer", [2]string{"Authorization", "Bearer xyz"}, "xyz"},
		{"none", [2]string{"", ""}, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodGet, "/metrics", nil)
			if tc.header[0] != "" {
				req.Header.Set(tc.header[0], tc.header[1])
			}
			if got := presentedKey(req); got != tc.want {
				t.Errorf("presentedKey = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestAuthMiddlewareBearer(t *testing.T) {
	withKeys := &server{apiKeys: parseAPIKeys([]string{"secret"})}
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/metrics", nil)
	req.Header.Set("Authorization", "Bearer secret")
	withKeys.authMiddleware(okHandler()).ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Errorf("status = %d, want 200 for Bearer auth", rec.Code)
	}
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
		trust      bool
		want       string
	}{
		{"remote addr with port", "1.2.3.4:5555", "", false, "1.2.3.4"},
		{"trusted x-forwarded-for single", "10.0.0.1:1", "203.0.113.5", true, "203.0.113.5"},
		{"trusted x-forwarded-for chain takes first", "10.0.0.1:1", "203.0.113.1, 70.41.3.18", true, "203.0.113.1"},
		{"untrusted x-forwarded-for is ignored", "10.0.0.1:1", "203.0.113.5", false, "10.0.0.1"},
		{"remote addr without port", "unixsocket", "", false, "unixsocket"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, "/score", nil)
			req.RemoteAddr = tc.remoteAddr
			if tc.xff != "" {
				req.Header.Set("X-Forwarded-For", tc.xff)
			}
			if got := clientIP(req, tc.trust); got != tc.want {
				t.Errorf("clientIP = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestClientIDFallsBackToIP(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/score", nil)
	req.RemoteAddr = "8.8.8.8:1234"
	if got := clientID(req, false); got != "ip:8.8.8.8" {
		t.Errorf("clientID = %q, want ip:8.8.8.8", got)
	}
}

func TestRateLimiterEviction(t *testing.T) {
	rl := newRateLimiter(1, 1)
	t0 := time.Unix(1_700_000_000, 0)
	rl.get("a", t0)
	if len(rl.clients) != 1 {
		t.Fatalf("clients = %d, want 1", len(rl.clients))
	}
	// A request past the idle TTL + sweep interval triggers a sweep that
	// evicts the now-idle "a" while registering the active "b".
	rl.get("b", t0.Add(limiterIdleTTL+2*limiterSweepEvery))
	if _, ok := rl.clients["a"]; ok {
		t.Error("idle client 'a' should have been evicted")
	}
	if _, ok := rl.clients["b"]; !ok {
		t.Error("active client 'b' should still be present")
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
