# Secret Management

Application secrets are stored in the managed secret store and injected at runtime. Secrets must never be committed to source control or placed in container images. Production database passwords rotate every 90 days, while short-lived cloud credentials expire within one hour. Suspected secret exposure requires immediate revocation.
