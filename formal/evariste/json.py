# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Any, Dict
from json import JSONEncoder
from pathlib import Path
import json
from json import decoder


class JSONPathEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Path):
            return {"__PATH__": str(obj)}
        return json.JSONEncoder.default(self, obj)


def load_path(dct):
    if "__PATH__" in dct:
        return Path(dct["__PATH__"])
    return dct


def dumps(obj: Any, *args, **kwargs) -> str:
    return json.dumps(obj, *args, cls=JSONPathEncoder, **kwargs)


def dump(obj: Any, *args, **kwargs) -> None:
    json.dump(obj, *args, cls=JSONPathEncoder, **kwargs)


def loads(s: str, *args, **kwargs) -> Dict:
    return json.loads(s, *args, object_hook=load_path, **kwargs)


def load(fp, *args, **kwargs) -> Dict:
    return json.load(fp, *args, object_hook=load_path, **kwargs)
