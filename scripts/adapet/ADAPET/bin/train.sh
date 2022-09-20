#!/usr/bin/env bash

set -exu

config_file=$1

python -m src.train -c $config_file
