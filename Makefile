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

# Release stuff

pre-release:
	python utils/release.py

pre-patch:
	python utils/release.py --patch

post-release:
	python utils/release.py --post_release

post-patch:
	python utils/release.py --post_release --patch

wheels:
	python setup.py bdist_wheel && python setup.py sdist

wheels_clean:
	rm -rf build && rm -rf dist

pypi_upload:
	python -m pip install twine
	twine upload dist/* -r pypi

pypi_test_upload:
	python -m pip install twine
	twine upload dist/* -r pypitest --repository-url=https://test.pypi.org/legacy/

pypi_test_install:
	python -m pip install evaluate==0.2.2 datasets==2.3.2 sentence_transformers==2.2.2
	python -m pip install -i https://testpypi.python.org/pypi setfit
	python -c "from setfit import *"
	echo "🚀 Successfully installed setfit from test.pypi.org"