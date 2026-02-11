import os

from lib.utils.tools.logger import Logger as Log
from . import base_evaluator

evaluators = {
    'base': base_evaluator.BaseEvaluator
}


def get_evaluator(configer, trainer, name=None):
    name = os.environ.get('evaluator', 'base')   # obtain env variables

    if not name in evaluators:
        raise RuntimeError('Unknown evaluator name: {}'.format(name))
    
    klass = evaluators[name]   # klass = BaseEvaluator
    Log.info('Using evaluator: {}'.format(klass.__name__))   

    return klass(configer, trainer)
