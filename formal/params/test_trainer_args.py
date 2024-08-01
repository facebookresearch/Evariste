# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import pytest
from dataclasses import dataclass

from params import Params, MISSING, ConfStore
from evariste.trainer.args import TrainerArgs
from params.test_params import check_reconstruct
from evariste.datasets.lean import LeanDatasetConf


@pytest.mark.slow
class TestParams:
    def test_n_layers(self):
        parser = TrainerArgs.to_cli()
        params = parser.parse_args(["--mm.dataset", "new3", "--model.n_layers", "6"])
        ta: TrainerArgs = TrainerArgs.from_cli(
            vars(params), default_instance=ConfStore["default_cfg"]
        )
        assert ta.model.enc_n_layers == -1
        ta.model.check_and_mutate_args()
        assert ta.model.enc_n_layers == 6
        assert ta.lean.dataset == LeanDatasetConf.from_dict(ta.lean.dataset.to_dict())
        check_reconstruct(ta)
        ta.check_type()


# This is a use case that appears in launcher/launch
# I parse the CLI once to produce some Params class
# Then, using these params, I mutate a whole part of the class, and re-parse to integrate these new-defaults.
@dataclass
class A(Params):
    a: int = 1


@dataclass
class B(Params):
    a: A = MISSING
    b: int = MISSING


default_b = B(A(a=1), b=MISSING)


class TestReload:
    def test_reload(self):
        parser = B.to_cli()
        params = parser.parse_args(["--b", "2"])
        b: B = B.from_cli(vars(params), default_instance=default_b)
        assert b == B(A(a=1), b=2)
        b.a.a = b.b
        parser = B.to_cli()
        params = parser.parse_args(["--b", "2"])
        b: B = B.from_cli(vars(params), default_instance=b)
        assert b == B(A(a=2), b=2)
