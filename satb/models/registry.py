""" Model Registry
Hacked together by / Copyright 2020 Ross Wightman

Modified by Lee Man.
NOTE: set up registry manully.
"""

from .dgdagrnn import DGDAGRNN
from .neurosat import NeuronSAT
from .deepsat import DeepSAT


# mli: current entrypoints are set up manully. Will modify to automatic registry later.
_model_entrypoints = {
    'neurosat': NeuronSAT,
    'dgdagrnn': DGDAGRNN,
    'deepsat': DeepSAT,
}


def is_model(model_name):
    """ Check if a model name exists
    """
    # return model_name in _model_entrypoints
    return model_name in _model_entrypoints


def model_entrypoint(model_name):
    """Fetch a model entrypoint for specified model name
    """
    return _model_entrypoints[model_name]