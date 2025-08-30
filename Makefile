.PHONY: install lint test train

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

lint:
	python -m pip install flake8 || true
	flake8 src tests || true

test:
	pytest -q

train:
	python -m src.train --data data/data.csv --out artifacts/
