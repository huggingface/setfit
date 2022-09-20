#!/usr/bin/env bash

set -exu

exp_dir=$1

python -m src.dev -e $exp_dir
