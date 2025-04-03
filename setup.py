# Lint as: python3
from pathlib import Path

from setuptools import find_packages, setup


README_TEXT = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

MAINTAINER = "Lewis Tunstall, Tom Aarsen"
MAINTAINER_EMAIL = "lewis@huggingface.co"

INTEGRATIONS_REQUIRE = ["optuna"]
REQUIRED_PKGS = [
    "datasets>=2.15.0",
    "sentence-transformers[train]>=3",
    "transformers>=4.41.0",
    "evaluate>=0.3.0",
    "huggingface_hub>=0.24.0",
    "scikit-learn",
    "packaging",
]
ABSA_REQUIRE = ["spacy<3.7.6"]
QUALITY_REQUIRE = ["black", "flake8", "isort", "tabulate"]
ONNX_REQUIRE = ["onnxruntime", "onnx!=1.16.2", "skl2onnx"]
OPENVINO_REQUIRE = ["hummingbird-ml", "openvino"]
TESTS_REQUIRE = ["pytest", "pytest-cov"] + ONNX_REQUIRE + OPENVINO_REQUIRE + ABSA_REQUIRE
DOCS_REQUIRE = ["hf-doc-builder>=0.3.0"]
CODECARBON_REQUIRE = ["codecarbon<2.6.0"]
# 2.7.* fails with AttributeError: 'EmissionsTracker' object has no attribute '_cloud'
# 2.6.* has an accidental print statement spamming the terminal
EXTRAS_REQUIRE = {
    "optuna": INTEGRATIONS_REQUIRE,
    "quality": QUALITY_REQUIRE,
    "tests": TESTS_REQUIRE,
    "onnx": ONNX_REQUIRE,
    "openvino": ONNX_REQUIRE + OPENVINO_REQUIRE,
    "docs": DOCS_REQUIRE,
    "absa": ABSA_REQUIRE,
    "codecarbon": CODECARBON_REQUIRE,
}


def combine_requirements(base_keys):
    return list(set(k for v in base_keys for k in EXTRAS_REQUIRE[v]))


EXTRAS_REQUIRE["dev"] = combine_requirements([k for k in EXTRAS_REQUIRE])
# For the combatibility tests we add pandas<2, as pandas 2.0.0 onwards is incompatible with old datasets versions,
# and we assume few to no users would use old datasets versions with new pandas versions.
# The only alternative is incrementing the minimum version for datasets, which seems unnecessary.
# Beyond that, fsspec is set to <2023.12.0 as that version is incompatible with datasets<=2.15.0
EXTRAS_REQUIRE["compat_tests"] = (
    [requirement.replace(">=", "==") for requirement in REQUIRED_PKGS]
    + TESTS_REQUIRE
    + ["pandas<2", "fsspec<2023.12.0"]
)

setup(
    name="setfit",
    version="1.1.2",
    description="Efficient few-shot learning with Sentence Transformers",
    long_description=README_TEXT,
    long_description_content_type="text/markdown",
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    url="https://github.com/huggingface/setfit",
    download_url="https://github.com/huggingface/setfit/tags",
    license="Apache 2.0",
    package_dir={"": "src"},
    packages=find_packages("src"),
    include_package_data=True,
    install_requires=REQUIRED_PKGS,
    extras_require=EXTRAS_REQUIRE,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="nlp, machine learning, fewshot learning, transformers",
    zip_safe=False,  # Required for mypy to find the py.typed file
)
