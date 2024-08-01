# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# from .env import EQVecBackwardEnv, EQEnvGenerator
# causes circular import -> model.data.env.equations imports .graph
# but on the way imports EQVecBackwardEnv which imports EquationEnv

from .graph import EQTactic, EQTheorem
