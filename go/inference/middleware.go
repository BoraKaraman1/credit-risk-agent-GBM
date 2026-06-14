package inference

import (
	"context"
	"crypto/subtle"
	"log/slog"
	"net"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"golang.org/x/time/rate"
)

type ctxKey int

const (
	loggerKey ctxKey = iota
	clientKey
)

// logger returns the request-scoped logger (carrying request_id), or
// the default logger outside an instrumented request.
func logger(ctx context.Context) *slog.Logger {
	if l, ok := ctx.Value(loggerKey).(*slog.Logger); ok {
		return l
	}
	return slog.Default()
}

// statusWriter captures the response status code for access logging and
// metrics. A handler that never calls WriteHeader implies 200.
type statusWriter struct {
	http.ResponseWriter
	status int
}

func (w *statusWriter) WriteHeader(code int) {
	w.status = code
	w.ResponseWriter.WriteHeader(code)
}

func (w *statusWriter) Write(b []byte) (int, error) {
	if w.status == 0 {
		w.status = http.StatusOK
	}
	return w.ResponseWriter.Write(b)
}

// instrument assigns a request ID, records HTTP metrics, and emits a
// structured access log line for one endpoint.
func instrument(endpoint string, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		id := r.Header.Get("X-Request-ID")
		if id == "" {
			id = uuid.NewString()
		}
		w.Header().Set("X-Request-ID", id)

		log := slog.With("request_id", id, "endpoint", endpoint)
		ctx := context.WithValue(r.Context(), loggerKey, log)

		sw := &statusWriter{ResponseWriter: w}
		start := time.Now()
		next.ServeHTTP(sw, r.WithContext(ctx))
		dur := time.Since(start)
		if sw.status == 0 {
			sw.status = http.StatusOK
		}

		httpRequests.WithLabelValues(endpoint, r.Method, strconv.Itoa(sw.status)).Inc()
		httpDuration.WithLabelValues(endpoint).Observe(dur.Seconds())
		log.Info("request",
			"method", r.Method, "status", sw.status,
			"duration_ms", float64(dur.Microseconds())/1000.0)
	})
}

// protect chains API-key authentication and per-client rate limiting in
// front of a handler.
func (s *server) protect(h http.Handler) http.Handler {
	return s.authMiddleware(s.rateLimitMiddleware(h))
}

// authMiddleware enforces a valid X-API-Key when keys are configured.
// With no keys configured it is a pass-through (auth disabled).
func (s *server) authMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if len(s.apiKeys) == 0 {
			next.ServeHTTP(w, r)
			return
		}
		presented := r.Header.Get("X-API-Key")
		if presented == "" || !keyAllowed(s.apiKeys, presented) {
			logger(r.Context()).Warn("unauthorized request", "remote", clientIP(r))
			writeJSON(w, http.StatusUnauthorized, map[string]string{"detail": "invalid or missing API key"})
			return
		}
		ctx := context.WithValue(r.Context(), clientKey, presented)
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

// keyAllowed compares the presented key against every configured key in
// constant time, without short-circuiting on a match.
func keyAllowed(keys [][]byte, presented string) bool {
	p := []byte(presented)
	var ok int
	for _, k := range keys {
		ok |= subtle.ConstantTimeCompare(k, p)
	}
	return ok == 1
}

// rateLimiter holds one token-bucket limiter per client identity.
type rateLimiter struct {
	mu      sync.Mutex
	clients map[string]*rate.Limiter
	rps     rate.Limit
	burst   int
}

func newRateLimiter(rps float64, burst int) *rateLimiter {
	return &rateLimiter{
		clients: make(map[string]*rate.Limiter),
		rps:     rate.Limit(rps),
		burst:   burst,
	}
}

// get returns the limiter for a client, creating it on first use. The
// map is unbounded in distinct client IPs; for this deployment the key
// space is small (a few integrators), so no eviction is needed.
func (rl *rateLimiter) get(client string) *rate.Limiter {
	rl.mu.Lock()
	defer rl.mu.Unlock()
	l, ok := rl.clients[client]
	if !ok {
		l = rate.NewLimiter(rl.rps, rl.burst)
		rl.clients[client] = l
	}
	return l
}

func (s *server) rateLimitMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		client := clientID(r)
		if !s.limiter.get(client).Allow() {
			logger(r.Context()).Warn("rate limit exceeded", "client", client)
			w.Header().Set("Retry-After", "1")
			writeJSON(w, http.StatusTooManyRequests, map[string]string{"detail": "rate limit exceeded"})
			return
		}
		next.ServeHTTP(w, r)
	})
}

// clientID identifies the caller for rate limiting: the authenticated
// API key when present, otherwise the client IP.
func clientID(r *http.Request) string {
	if k, ok := r.Context().Value(clientKey).(string); ok && k != "" {
		return "key:" + k
	}
	return "ip:" + clientIP(r)
}

// clientIP extracts the caller's IP, honoring the first X-Forwarded-For
// hop when present (the API runs behind compose / a reverse proxy).
func clientIP(r *http.Request) string {
	if xff := r.Header.Get("X-Forwarded-For"); xff != "" {
		if i := strings.IndexByte(xff, ','); i >= 0 {
			return strings.TrimSpace(xff[:i])
		}
		return strings.TrimSpace(xff)
	}
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		return r.RemoteAddr
	}
	return host
}
