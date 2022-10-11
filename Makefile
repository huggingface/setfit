# make sure to test the local checkout in scripts and not the pre-installed one (don't use quotes!)
export PYTHONPATH = src

check_dirs := scripts tests src

style:
	python -m black --line-length 119 --exclude="scripts/adapet|scripts/tfew" --target-version py39 $(check_dirs)
	python -m isort --skip scripts/adapet --skip scripts/tfew $(check_dirs)

quality:
	python -m black --check --line-length 119 --exclude="scripts/adapet|scripts/tfew" --target-version py39 $(check_dirs)
	python -m isort --check-only --skip scripts/adapet --skip scripts/tfew $(check_dirs)
	python -m flake8 --max-line-length 119 $(check_dirs)

test:
	python -m pytest -sv tests/

coverage:
	python -m pytest --cov=src --cov-report=term-missing -sv tests/
