#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from morl.utils.utils import _HAS_FLUID, _HAS_PADDLE, _HAS_TORCH
from morl.utils import logger

if 'morl_BACKEND' in os.environ and os.environ['morl_BACKEND'] != '':
    assert os.environ['morl_BACKEND'] in ['fluid', 'paddle', 'torch']
    if os.environ['morl_BACKEND'] == 'paddle':
        from morl.algorithms.paddle import *
    elif os.environ['morl_BACKEND'] == 'torch':
        assert _HAS_TORCH, 'Torch-based morl requires torch, which is not installed.'
        from morl.algorithms.torch import *
else:
    if _HAS_PADDLE:
        from morl.algorithms.paddle import *
    elif _HAS_TORCH:
        from morl.algorithms.torch import *
    else:
        logger.warning(
            "No deep learning framework was found, but it's ok for parallel computation."
        )
