# Lint as: python3
"""Setup for repository"""

from setuptools import find_packages, setup


DOCLINES = __doc__.split("\n")

REQUIRED_PKGS = ["datasets==2.3.2", "sentence-transformers==2.2.2", "evaluate==0.2.2"]

QUALITY_REQUIRE = ["black", "flake8", "isort"]

TESTS_REQUIRE = ["pytest", "pytest-cov"]

EXTRAS_REQUIRE = {"quality": QUALITY_REQUIRE, "tests": TESTS_REQUIRE}


def combine_requirements(base_keys):
    return list(set(k for v in base_keys for k in EXTRAS_REQUIRE[v]))


EXTRAS_REQUIRE["dev"] = combine_requirements([k for k in EXTRAS_REQUIRE])


setup(
    name="setfit",
    version="0.1.1",
    description=DOCLINES[0],
    long_description="\n".join(DOCLINES[2:]),
    author="SetFit",
    author_email="lewis@huggingface.co",
    url="https://github.com/SetFit/setfit",
    download_url="https://github.com/SetFit/setfit/tags",
    license="Apache 2.0",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=REQUIRED_PKGS,
    extras_require=EXTRAS_REQUIRE,
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="nlp, machine learning, fewshot learning, transformers",
    zip_safe=False,  # Required for mypy to find the py.typed file
)
