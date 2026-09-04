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
    # evaluate < 0.4.6 breaks metric loading with huggingface_hub v1.0
    "evaluate>=0.4.6",
    "huggingface_hub>=0.24.0",
    "scikit-learn",
    "packaging",
]
# spaCy 3.8 depends on blis releases without Python 3.9 wheels, and the spaCy 3.7 binaries need numpy 1.x
ABSA_REQUIRE = [
    "spacy<3.8; python_version < '3.10'",
    "numpy<2; python_version < '3.10'",
    "spacy; python_version >= '3.10'",
]
QUALITY_REQUIRE = ["black", "flake8", "isort", "tabulate"]
# The Python 3.9 wheels of onnx 1.17 to 1.19 crash on Windows once pyarrow.dataset has been imported
ONNX_REQUIRE = ["onnxruntime", "onnx!=1.16.2", "onnx<1.17; python_version < '3.10'", "skl2onnx"]
# hummingbird-ml pins onnx<=1.16.1, which has no wheels for Python 3.13
OPENVINO_REQUIRE = ["hummingbird-ml", "openvino"]
TESTS_REQUIRE = ["pytest", "pytest-cov"] + ONNX_REQUIRE + ABSA_REQUIRE
DOCS_REQUIRE = ["hf-doc-builder>=0.3.0"]
# 2.6.* has an accidental print statement spamming the terminal
# 2.7.* and 2.8.* lock out a second EmissionsTracker in the same process (allow_multiple_runs only
# defaults to True from 3.0.0), and the locked-out tracker then fails with a missing _cloud
CODECARBON_REQUIRE = ["codecarbon!=2.6.*,!=2.7.*,!=2.8.*"]
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
# Beyond that, fsspec is set to <2023.12.0 as that version is incompatible with datasets<=2.15.0,
# and numpy<2 as pandas<2 cannot be imported with numpy 2.
EXTRAS_REQUIRE["compat_tests"] = (
    [requirement.replace(">=", "==") for requirement in REQUIRED_PKGS]
    + TESTS_REQUIRE
    + ["pandas<2", "fsspec<2023.12.0", "numpy<2"]
)
# The last dependency generation before transformers v5, huggingface_hub v1 and Sentence Transformers v6
EXTRAS_REQUIRE["compat_tests_v4"] = TESTS_REQUIRE + [
    "transformers<5",
    "sentence-transformers[train]<6",
    "huggingface_hub<1",
    "datasets<5",
]

setup(
    name="setfit",
    version="1.2.0",
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
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    keywords="nlp, machine learning, fewshot learning, transformers",
    zip_safe=False,  # Required for mypy to find the py.typed file
)
