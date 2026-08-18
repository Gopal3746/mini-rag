.PHONY: setup up down logs test lint verify qa ingest query evaluate reset

setup:
	cp -n .env.example .env || true
	mkdir -p data

up: setup
	docker compose up --build -d

down:
	docker compose down

logs:
	docker compose logs -f api worker

test:
	docker compose run --rm --no-deps api pytest /app/tests

lint:
	docker compose run --rm --no-deps api ruff check /app/src /app/tests

verify:
	python3 scripts/verify_project.py

qa: verify lint test

ingest:
	docker compose exec api rag-ingest ingest /data/sample_docs --wait

query:
	docker compose exec api rag-ingest query "How long are production database backups retained?" --retrieve-only

evaluate:
	docker compose exec api rag-ingest evaluate --config /app/configs/experiments.yaml

reset:
	docker compose down -v
