#!/bin/bash

if [ ! -d "data/superglue/" ] ; then
    mkdir -p data
    cd data
    mkdir -p superglue
    cd superglue
    wget "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/combined.zip"
    unzip combined.zip
    cd ../..
fi

if [ ! -d "data/fewglue/" ] ; then
    mkdir -p data
    cd data
    git clone https://github.com/timoschick/fewglue.git
    cd fewglue
    rm -rf .git
    rm README.md
    mv FewGLUE/* .
    rm -r FewGLUE
    cd ../..
fi

if [ ! -d "env" ] ; then
    python -m venv env
    source env/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
fi