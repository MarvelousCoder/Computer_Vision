import os
from utils import *
from pathlib import Path

DIRNAME = os.path.dirname(__file__)
DATA_PATH = os.path.join(DIRNAME, "../data/lfw-deepfunneled")

people_dict = Path(os.path.join(DATA_PATH, "../people_dict.pkl"))