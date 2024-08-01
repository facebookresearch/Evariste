# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from numbers import Number
from typing import Dict


def prefix_dict(src: Dict[str, Number], prefix: str) -> Dict[str, Number]:
    return {f"{prefix}{k}": v for k, v in src.items()}
