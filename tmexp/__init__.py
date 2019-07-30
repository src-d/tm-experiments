# Imports below are necessary to register the commands
# flake8: noqa
from .create_bow import create_bow
from .label import label
from .merge import merge
from .metrics import compute_metrics
from .postprocess import postprocess
from .preprocess import preprocess
from .train_artm import train_artm
from .train_hdp import train_hdp
from .visualize import visualize

__version__ = "0.1.0"
