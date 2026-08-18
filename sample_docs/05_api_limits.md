# Public API Rate Limits

Authenticated API clients are limited to 600 requests per minute per organization by default. Burst traffic may use up to 100 requests in a 10-second window. The API returns HTTP 429 when the limit is exceeded and includes a Retry-After header. Enterprise customers may request higher contractual limits.
