# Database Schema Changes

Schema migrations use an expand-and-contract process. Backward-compatible columns or tables are added first, application code is deployed second, and destructive changes occur only after old code paths are removed. Long-running table rewrites require a reviewed migration plan and maintenance-window approval.
