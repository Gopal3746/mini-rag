# Production Database Backup Policy

Production PostgreSQL databases receive encrypted full backups every 24 hours and transaction-log backups every 15 minutes. Daily backups are retained for 35 days. A monthly backup is retained for 12 months. Restore drills are performed quarterly in a staging environment. Backup encryption keys are managed separately from database credentials.
