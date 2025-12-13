.PHONY: research.install dev.install


dev.install:
	poetry install --with dev

research.install:
	poetry install --with research



lint:
	@isort ./service/ ./tests/  --settings-file ./setup.cfg
	@black ./service/ ./tests/ --config pyproject.toml
	@flake8 --config ./setup.cfg ./service/
	@flake8 --config ./flake8.tests.ini ./tests/
	@mypy ./service/ --ignore-missing-imports --config-file setup.cfg



dc.up:
	docker comnpose up -d

dc.down:
	docker compose down


run:
	python -m service
