# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import pytest
from typing import Optional
from dataclasses import dataclass, asdict, field
from params import Params, ConfStore, MISSING, OptionalDataClass, MissingArg


@dataclass
class A(Params):
    a1: int = MISSING
    a2: Optional[int] = MISSING
    a3: Optional[int] = None
    a4: Optional[int] = 42


ConfStore["a_def"] = A(a1=1, a2=1)


@dataclass
class B(Params):
    b1: A = field(default_factory=lambda: A(a1=2, a2=2))
    b2: Optional[A] = None


parser = A.to_cli()


class TestParams:
    def test_A_1(self):
        params = parser.parse_args(["--a1", "1", "--a2", "1", "--a3", "1"])
        a = A.from_cli(vars(params), default_instance=A())
        assert a == A(a1=1, a2=1, a3=1)

    def test_A_2(self):
        params = parser.parse_args(["--a1", "1", "--a2", "1"])
        a = A.from_cli(vars(params), default_instance=A())
        assert a == A(a1=1, a2=1)

    #
    def test_A_missing(self):
        params = parser.parse_args(["--a1", "1"])
        with pytest.raises(MissingArg):
            a = A.from_cli(vars(params), default_instance=A())

    def test_B2_none(self):
        params = B.to_cli().parse_args([])
        b = B.from_cli(vars(params), default_instance=B())
        assert b == B()

    def test_B2_some_but_missing(self):
        params = B.to_cli().parse_args(["--b2.a1", "1"])
        with pytest.raises(MissingArg):
            b = B.from_cli(vars(params), default_instance=B())
            assert b == B(b2=A(a1=1))

    def test_B2_some(self):
        params = B.to_cli().parse_args(["--b2.a1", "1", "--b2.a2", "5"])
        b = B.from_cli(vars(params), default_instance=B())
        assert b == B(b2=A(a1=1, a2=5))
