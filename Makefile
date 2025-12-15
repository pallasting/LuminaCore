.PHONY: install test build-frontend build-python clean all format lint type-check quality

install:
	pip install -e .
	cd frontend && npm install

test:
	python -m pytest tests/

build-frontend:
	cd frontend && npm run build

build-python:
	python -m build

clean:
	rm -rf dist/ build/ *.egg-info/ frontend/dist/

format:
	black lumina/ tests/
	isort lumina/ tests/

lint:
	flake8 lumina/ tests/

type-check:
	mypy lumina/ tests/

quality: format lint type-check
	@echo "Code quality checks completed!"

all: install test build-frontend build-python quality