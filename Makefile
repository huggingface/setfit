# make sure to test the local checkout in scripts and not the pre-installed one (don't use quotes!)
export PYTHONPATH = src

check_dirs := scripts tests src

style:
	python -m black --line-length 119 --target-version py38 $(check_dirs)
	python -m isort $(check_dirs)

quality:
	python -m black --check --line-length 119 --target-version py38 $(check_dirs)
	python -m isort --check-only $(check_dirs)
	python -m flake8 --max-line-length 119 --exclude=results $(check_dirs)

test:
	python -m pytest -sv tests/