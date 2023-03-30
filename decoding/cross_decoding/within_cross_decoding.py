"""
Mimics the cross-decoding analysis but within each session. 
Splits each session up into 11 parts, and trains and tests across all pairs of parts.


This is done in sensor space. 
"""

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
import numpy as np
import os
import multiprocessing as mp
from time import perf_counter
import argparse

# local imports
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parents[2])) # adds the parent directory to the path so that the utils module can be imported

from utils.data.concatenate import flip_sign, read_and_concate_sessions_source, read_and_concate_sessions
from utils.data.triggers import get_triggers_equal, convert_triggers_animate_inanimate, balance_class_weights
from utils.analysis.decoder import Decoder
