.PHONY: help install lint type test test-int test-scenarios eval benchmark ingest train up down

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "%-20s %s\n", $$1, $$2}'

install: ## Install all dependencies
	pip install -e ".[dev,eval,api,braintrust]"
	pre-commit install

up: ## Start docker-compose services
	docker compose -f infra/docker-compose.yml up -d

down: ## Stop docker-compose services
	docker compose -f infra/docker-compose.yml down

lint: ## Run ruff linter
	ruff check src/ tests/ scripts/
	ruff format --check src/ tests/ scripts/

type: ## Run mypy type checker
	mypy src/ scripts/

test: ## Run unit tests
	pytest tests/unit/ -v --cov=src/knowledge_engine --cov-report=term-missing

test-int: ## Run integration tests (requires: make up)
	pytest tests/integration/ -v -m integration

test-scenarios: ## Run use case scenario tests
	pytest tests/integration/ tests/eval/ -v -m use_case

ingest: ## Fetch and ingest EDGAR filings
	python -m knowledge_engine.ingestion.edgar_client

train: ## Run LoRA fine-tuning (requires ENABLE_GPU_OPS=1)
	python -m knowledge_engine.training.lora_trainer

eval: ## Run RAGAS evaluation on gold dataset
	pytest tests/eval/ -v -m "not gpu"

benchmark: ## Run A/B benchmark (Sonnet vs. SLM)
	python scripts/run_ab_benchmark.py
