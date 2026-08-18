# Observability Standards

Every production service publishes request rate, error rate, and latency metrics. Distributed traces propagate a trace identifier across service boundaries. Application logs must be structured JSON and must not contain secrets, authentication tokens, or full payment card data. Paging alerts should represent actionable user impact rather than raw resource utilization.
