# Deployment and Rollback Guide

Services deploy through the CI pipeline after tests and policy checks pass. Production deployments use canary rollout beginning at 5 percent of traffic, then 25 percent, then 100 percent if error-rate and latency gates remain healthy. Every service must expose a rollback command. A failed health gate automatically stops promotion.
