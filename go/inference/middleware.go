package inference

import (
	"context"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/hex"
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
	requestIDKey
)

// logger returns the request-scoped logger (carrying request_id), or
// the default logger outside an instrumented request.
func logger(ctx context.Context) *slog.Logger {
	if l, ok := ctx.Value(loggerKey).(*slog.Logger); ok {
		return l
	}
	return slog.Default()
}

func requestID(ctx context.Context) string {
	id, _ := ctx.Value(requestIDKey).(string)
	return id
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
		ctx = context.WithValue(ctx, requestIDKey, id)

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

// protect chains API-key authentication and request rate limiting in
// front of a handler. Scoring operations have their own weighted limiter.
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
		presented := presentedKey(r)
		if presented == "" || !keyAllowed(s.apiKeys, presented) {
			logger(r.Context()).Warn("unauthorized request", "remote", clientIP(r, s.trustProxy))
			writeJSON(w, http.StatusUnauthorized, map[string]string{"detail": "invalid or missing API key"})
			return
		}
		ctx := context.WithValue(r.Context(), clientKey, presented)
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

// presentedKey extracts the caller's API key from either the X-API-Key
// header or an "Authorization: Bearer <key>" header. The Bearer form lets
// standard scrapers (e.g. Prometheus) authenticate to /metrics.
func presentedKey(r *http.Request) string {
	if k := r.Header.Get("X-API-Key"); k != "" {
		return k
	}
	const prefix = "Bearer "
	if a := r.Header.Get("Authorization"); strings.HasPrefix(a, prefix) {
		return strings.TrimSpace(a[len(prefix):])
	}
	return ""
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

// rate-limiter eviction: idle client buckets are swept so the map cannot
// grow without bound (e.g. under many distinct source IPs).
const (
	limiterIdleTTL    = 10 * time.Minute
	limiterSweepEvery = 1 * time.Minute
)

// clientLimiter is one client's token bucket plus its last-seen time.
type clientLimiter struct {
	limiter  *rate.Limiter
	lastSeen time.Time
}

// rateLimiter holds one token-bucket limiter per client identity.
type rateLimiter struct {
	mu        sync.Mutex
	clients   map[string]*clientLimiter
	rps       rate.Limit
	burst     int
	lastSweep time.Time
}

func newRateLimiter(rps float64, burst int) *rateLimiter {
	return &rateLimiter{
		clients: make(map[string]*clientLimiter),
		rps:     rate.Limit(rps),
		burst:   burst,
	}
}

// get returns the limiter for a client, creating it on first use and
// opportunistically evicting buckets idle longer than limiterIdleTTL so
// the map stays bounded. The sweep runs at most once per limiterSweepEvery.
func (rl *rateLimiter) get(client string, now time.Time) *rate.Limiter {
	rl.mu.Lock()
	defer rl.mu.Unlock()
	if now.Sub(rl.lastSweep) > limiterSweepEvery {
		for k, cl := range rl.clients {
			if now.Sub(cl.lastSeen) > limiterIdleTTL {
				delete(rl.clients, k)
			}
		}
		rl.lastSweep = now
	}
	cl, ok := rl.clients[client]
	if !ok {
		cl = &clientLimiter{limiter: rate.NewLimiter(rl.rps, rl.burst)}
		rl.clients[client] = cl
	}
	cl.lastSeen = now
	return cl.limiter
}

func (s *server) rateLimitMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		client := clientID(r, s.trustProxy)
		if s.requestLimiter == nil {
			next.ServeHTTP(w, r)
			return
		}
		if !s.requestLimiter.get(client, time.Now()).Allow() {
			logger(r.Context()).Warn("rate limit exceeded", "client", client)
			w.Header().Set("Retry-After", "1")
			writeJSON(w, http.StatusTooManyRequests, map[string]string{"detail": "rate limit exceeded"})
			return
		}
		next.ServeHTTP(w, r)
	})
}

// clientID identifies the caller for rate limiting: a short digest of
// the authenticated API key when present, otherwise the client IP. The
// digest — never the key itself — is what reaches limiter state and
// log lines, so a 429 cannot leak a live credential into access logs.
func clientID(r *http.Request, trustForwarded bool) string {
	if k, ok := r.Context().Value(clientKey).(string); ok && k != "" {
		sum := sha256.Sum256([]byte(k))
		return "key:" + hex.EncodeToString(sum[:4])
	}
	return "ip:" + clientIP(r, trustForwarded)
}

// clientIP extracts the caller's IP. The X-Forwarded-For header is honored
// only when trustForwarded is set (the API sits behind a trusted proxy);
// otherwise it is ignored so a direct client cannot spoof its identity to
// evade rate limiting or poison access logs.
func clientIP(r *http.Request, trustForwarded bool) string {
	if trustForwarded {
		if xff := r.Header.Get("X-Forwarded-For"); xff != "" {
			if i := strings.IndexByte(xff, ','); i >= 0 {
				return strings.TrimSpace(xff[:i])
			}
			return strings.TrimSpace(xff)
		}
	}
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		return r.RemoteAddr
	}
	return host
}
