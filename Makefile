.PHONY: research.install dev.install

args := $(wordlist 2, 100, $(MAKECMDGOALS))
ifndef args
MESSAGE = "No such command (or you pass two or many targets to ). List of possible commands: make help"
else
MESSAGE = "Done"
endif


env:
	cp .env.default .env


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
	docker compose up -d

dc.down:
	docker compose down


# Migrations

db.revision:
	cd service/db && alembic revision --autogenerate -m $(args)

db.migrate:
	cd service/db && alembic upgrade head

db.rollback:
	cd service/db && alembic downgrade -1


run:
	python -m service
