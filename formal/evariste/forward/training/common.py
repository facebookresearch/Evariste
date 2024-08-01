# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import TypeVar

from evariste.backward.graph import Theorem, Tactic


SomeTheorem = TypeVar("SomeTheorem", bound=Theorem)
SomeTactic = TypeVar("SomeTactic", bound=Tactic)
