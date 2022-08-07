import argparse
import json
import os
import tarfile
from glob import glob
from os.path import splitext
from typing import List, Tuple

from numpy import mean, std


# INPUT: t011b_pretrained_${dataset}/train-${sample_size}-${split0-9}_seed${0-4}
# OUTPUT: t011b_pretrained100k_${dataset}_seed${seed}_ia3


def get_sample_sizes(path: str) -> List[str]:
    return sorted(list({int(name.split("-")[-2]) for name in glob(f"{path}/train-*-0")}))


get_sample_sizes("/home/daniel_nlp/setfit/scripts/tfew/results/t03b_pretrained_emotion")
