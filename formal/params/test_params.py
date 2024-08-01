# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass, field
from typing import Optional, List, Dict
from argparse import Namespace
from pathlib import Path
import pytest

from params import (
    Params,
    ConfStore,
    MISSING,
    MissingArg,
    WrongConfName,
    WrongArgType,
    WrongConfType,
    WrongFieldType,
    DefaultDataClassValue,
    NOCLI,
)


@dataclass
class AB(Params):
    a: int = MISSING
    b: int = MISSING


ConfStore["ab_1"] = AB(a=1, b=1)


@dataclass
class C(Params):
    ab: AB = field(default_factory=lambda: AB(a=3, b=3))
    c: int = MISSING


ConfStore["c_1"] = C(c=1)


@dataclass
class D(Params):
    c: C = field(default_factory=lambda: C(c=4))
    d: int = MISSING


parser = D.to_cli()


@dataclass
class C2(Params):
    ab: AB = MISSING
    c: int = MISSING


@dataclass
class D2(Params):
    c: C2 = MISSING
    d: int = MISSING


@dataclass
class D3(Params):
    a: int
    b: Optional[int]
    c: float = 2.0
    d: List[int] = field(default_factory=lambda: [2, 3])


@dataclass
class D4(Params):
    a: int
    b: int = NOCLI()


parser2 = D2.to_cli()


def check_reconstruct(x: Params):
    cls = x.__class__
    assert x == cls.from_dict(x.to_dict())
    assert x == cls.from_json(x.to_json())
    assert x == cls.from_flat(x.to_flat())


class TestParams:
    def test_simple(self):
        params = parser.parse_args(
            ["--d", "1", "--c.c", "2", "--c.ab.a", "3", "--c.ab.b", "4"]
        )
        d = D.from_cli(vars(params), default_instance=D())
        assert d == D(d=1, c=C(c=2, ab=AB(a=3, b=4))), d

    def test_simple_repeated(self):
        params = parser.parse_args(
            [
                "--d",
                "1",
                "--c.c",
                "2",
                "--c.ab.a",
                "3",
                "--c.ab.b",
                "4",
                "--c.ab.a",
                "4",
            ]
        )
        d = D.from_cli(vars(params), default_instance=D())
        assert d == D(d=1, c=C(c=2, ab=AB(a=4, b=4))), d

    def test_conf_store(self):
        params = parser.parse_args(["--d", "1", "--c", "c_1"])
        d = D.from_cli(vars(params), default_instance=D())
        assert d == D(d=1, c=ConfStore["c_1"]), d

    def test_conf_store_replace(self):
        params = parser.parse_args(["--d", "1", "--c", "c_1", "--c.ab", "ab_1"])
        d = D.from_cli(vars(params), default_instance=D())
        assert d == D(d=1, c=C(c=1, ab=ConfStore["ab_1"])), d

    def test_conf_store_deep(self):
        params = parser.parse_args(["--d", "1", "--c.c", "3", "--c.ab", "ab_1"])
        d = D.from_cli(vars(params), default_instance=D())
        assert d == D(d=1, c=C(c=3, ab=ConfStore["ab_1"])), d

    def test_not_in_conf_store(self):
        params = parser.parse_args(["--d", "1", "--c.c", "3", "--c.ab", "ab_3"])
        with pytest.raises(WrongConfName):
            D.from_cli(vars(params), default_instance=D())

    def test_missing_field(self):
        params = parser.parse_args(["--c.c", "3", "--c.ab", "ab_1"])
        with pytest.raises(MissingArg):
            D.from_cli(vars(params), default_instance=D())

    def test_wrong_arg_type(self):
        params = Namespace(
            **{
                "d": 1,
                "c.c": "lol",
                "c.ab": "ab_1",
                "c": MISSING,
                "c.ab.a": MISSING,
                "c.ab.b": MISSING,
            }
        )
        with pytest.raises(WrongArgType):
            D.from_cli(vars(params), default_instance=D())

    def test_wrong_arg_type_class(self):
        params = Namespace(
            **{
                "d": 1,
                "c": 2,
                "c.ab": "ab_1",
                "c.c": MISSING,
                "c.ab.a": MISSING,
                "c.ab.b": MISSING,
            }
        )
        with pytest.raises(WrongArgType):
            D.from_cli(vars(params), default_instance=D())

    def test_wrong_conf_type(self):
        params = parser.parse_args(["--c.c", "3", "--c.ab", "c_1"])
        with pytest.raises(WrongConfType):
            D.from_cli(vars(params), default_instance=D())

    def test_incomplete_confstore_completed(self):
        if "ab_2" in ConfStore:
            del ConfStore["ab_2"]
        ConfStore["ab_2"] = AB(a=2, b=MISSING)
        params = parser.parse_args(
            ["--d", "1", "--c.c", "3", "--c.ab", "ab_2", "--c.ab.b", "2"]
        )
        d = D.from_cli(vars(params), default_instance=D())
        assert d == D(d=1, c=C(ab=AB(a=2, b=2), c=3)), d

    def test_incomplete_confstore_incomplete(self):
        if "ab_2" in ConfStore:
            del ConfStore["ab_2"]
        ConfStore["ab_2"] = AB(a=2, b=MISSING)
        params = parser.parse_args(["--d", "1", "--c.c", "3", "--c.ab", "ab_2"])
        with pytest.raises(MissingArg):
            D.from_cli(vars(params), default_instance=D())

    def test_incomplete_confstore_completed_2(self):
        if "c_2" in ConfStore:
            del ConfStore["c_2"]
        ConfStore["c_2"] = C(c=4, ab=MISSING)
        params = parser.parse_args(["--d", "1", "--c", "c_2", "--c.ab", "ab_1"])
        d = D.from_cli(vars(params), default_instance=D())
        assert d == D(d=1, c=C(ab=ConfStore["ab_1"], c=4)), d

    def test_incomplete_confstore_completed2_2(self):
        if "c_2" in ConfStore:
            del ConfStore["c_2"]
        ConfStore["c_2"] = C2(c=4, ab=MISSING)
        params = parser2.parse_args(
            ["--d", "1", "--c", "c_2", "--c.ab.a", "5", "--c.ab.b", "5"]
        )
        d = D2.from_cli(vars(params), default_instance=D2())
        assert d == D2(d=1, c=C2(ab=AB(a=5, b=5), c=4)), d

    def test_incomplete_confstore_incomplete_2(self):
        if "c_2" in ConfStore:
            del ConfStore["c_2"]
        ConfStore["c_2"] = C2(c=4, ab=MISSING)
        params = parser2.parse_args(["--d", "1", "--c", "c_2"])
        with pytest.raises(MissingArg):
            D2.from_cli(vars(params), default_instance=D2())

    def test_incomplete_confstore_incomplete2_2(self):
        if "c_2" in ConfStore:
            del ConfStore["c_2"]
        ConfStore["c_2"] = C2(c=4, ab=MISSING)
        params = parser2.parse_args(["--d", "1", "--c", "c_2", "--c.ab.a", "5"])
        with pytest.raises(MissingArg):
            D2.from_cli(vars(params), default_instance=D2())

    def test_missing(self):
        xx = [
            (C(c=4, ab=MISSING), {"ab"}),
            (C(c=4, ab=AB(a=1, b=3)), set()),
            (AB(), {"a", "b"}),
            (C(), {"c"}),
            (C2(ab=AB(a=5, b=5)), {"c"}),
            (C2(ab=AB(a=5, b=5), c=4), set()),
            (D2(d=1, c=C2(ab=AB(a=5, b=5), c=4)), set()),
        ]
        for x, missing in xx:
            assert x.has_missing() == (len(missing) > 0)
            assert x.get_missing() == missing

    def test_reconstruct(self):
        check_reconstruct(AB(a=-1, b=2))
        check_reconstruct(C(c=4, ab=AB(a=1, b=3)))
        check_reconstruct(D(d=1, c=C(c=3, ab=ConfStore["ab_1"])))

    def test_check_type(self):
        xx = [
            (D3(a=1, b=2), True),
            (D3(a=1, b=2, c=3), True),
            (D3(a=1, b=2, c=3.0), True),
            (D3(a=1.0, b=2, c=3), False),
            (D3(a=1, b=None), True),
            (D3(a=None, b=None), False),
            (D3(a=1, b="2"), False),
            (D3(a=1, b=D3(a=1, b=2)), False),
            (D3(a=1, b=None), True),
            (D3(a=1, b=None, d=[2]), True),
            (D3(a=1, b=None, d=[2.2]), False),
        ]
        for x, is_valid in xx:
            if is_valid:
                x.check_type()
            else:
                with pytest.raises(WrongFieldType):
                    x.check_type()

    def test_nocli(self):
        _, extra1 = D4.to_cli().parse_known_args(["--a", "1"])
        _, extra2 = D4.to_cli().parse_known_args(["--a", "1", "BLA"])
        _, extra3 = D4.to_cli().parse_known_args(["--a", "1", "--b", "2"])
        assert extra1 == []
        assert extra2 == ["BLA"]
        assert extra3 == ["--b", "2"]

    def test_nodict(self):
        @dataclass
        class D5(Params):
            a: Dict[int, int]

        with pytest.raises(RuntimeError):
            D5({1: 1})

    def test_no_default_args(self):
        @dataclass
        class CC1(Params):
            ab: AB = field(default_factory=lambda: AB(a=3, b=3))
            c: int = MISSING

        CC1()

        with pytest.raises(DefaultDataClassValue):

            @dataclass
            class CC2(Params):
                ab: AB = AB(a=3, b=3)
                c: int = MISSING

            CC2()

            @dataclass
            class CC3(Params):
                ab: AB = field(default_factory=2)
                c: int = MISSING

            CC3()

            @dataclass
            class CC4(Params):
                ab: AB = field(default_factory=lambda: D())
                c: int = MISSING

            CC4()

    def test_path_json(self):
        @dataclass
        class E(Params):
            a: Path

        e = E(a=Path("/a"))
        assert e == E.from_json(e.to_json())

    def test_path_cli(self):
        @dataclass
        class E(Params):
            a: Path

        e = E(a=Path("/a"))
        parsed = E.to_cli().parse_args(["--a", "/a"])
        e2 = E.from_cli(vars(parsed))
        assert e == e2

    def test_from_flat(self):
        c = C.from_flat({"c": 0})
        assert c == C(ab=AB(a=3, b=3), c=0)
