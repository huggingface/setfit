# Lint as: python3
from pathlib import Path

from setuptools import find_packages, setup


README_TEXT = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

INTEGRATIONS_REQUIRE = ["optuna"]

REQUIRED_PKGS = ["datasets>=2.3.0", "sentence-transformers>=2.2.1", "evaluate>=0.3.0"]

QUALITY_REQUIRE = ["black", "flake8", "isort", "tabulate"]

ONNX_REQUIRE = ["onnxruntime", "onnx", "skl2onnx"]

TESTS_REQUIRE = ["pytest", "pytest-cov"] + ONNX_REQUIRE

COMPAT_TESTS_REQUIRE = [requirement.replace(">=", "==") for requirement in REQUIRED_PKGS] + TESTS_REQUIRE

EXTRAS_REQUIRE = {
    "optuna": INTEGRATIONS_REQUIRE,
    "quality": QUALITY_REQUIRE,
    "tests": TESTS_REQUIRE,
    "onnx": ONNX_REQUIRE,
    "compat_tests": COMPAT_TESTS_REQUIRE,
}


def combine_requirements(base_keys):
    return list(set(k for v in base_keys for k in EXTRAS_REQUIRE[v]))


EXTRAS_REQUIRE["dev"] = combine_requirements([k for k in EXTRAS_REQUIRE])


setup(
    name="setfit",
    version="0.5.0",
    description="Efficient few-shot learning with Sentence Transformers",
    long_description=README_TEXT,
    long_description_content_type="text/markdown",
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
