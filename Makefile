.PHONY: research.install dev.install


env:
	uv venv

research.install:
	uv sync --group research


dev.install:
	uv sync --group dev


dc.up:
	@docker compose up -d


dc.down:
	@docker compose down