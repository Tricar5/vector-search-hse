.PHONY: research.install dev.install


env:
	uv venv

research.install:
	uv sync --group research


dev.install:
	uv sync --group dev


lint:
	ruff check ./service --fix

format:
	ruff format ./service


run:
	uvicorn