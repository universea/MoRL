
__version__ = "0.0.1"
"""
generates new MORL python API
"""
import os

from tensorboardX import SummaryWriter
from morl.utils.utils import _HAS_FLUID, _HAS_TORCH, _HAS_PADDLE
from morl.utils import logger

if 'morl_BACKEND' in os.environ and os.environ['morl_BACKEND'] != '':
    assert os.environ['morl_BACKEND'] in ['fluid', 'paddle', 'torch']
    logger.info(
        'Have found environment variable `morl_BACKEND`==\'{}\', switching backend framework to [{}]'
        .format(os.environ['morl_BACKEND'], os.environ['morl_BACKEND']))
    if os.environ['morl_BACKEND'] == 'paddle':
        from morl.core.paddle import *
    elif os.environ['morl_BACKEND'] == 'torch':
        assert _HAS_TORCH, 'Torch-based morl requires torch, which is not installed.'
        from morl.core.torch import *
else:
    if _HAS_PADDLE:
        from morl.core.paddle import *
    elif _HAS_TORCH:
        from morl.core.torch import *

#from morl.remote import remote_class, connect
from morl import algorithms
