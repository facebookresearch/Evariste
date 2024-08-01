# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from abc import ABC, abstractmethod
from typing import Tuple, List
import re
import math
import numpy as np


NAN_TOKEN = "<NAN>"
INF_TOKEN = "<INF>"
NEG_INF_TOKEN = "<-INF>"


class FloatTokenError(Exception):
    pass


class Tokenizer(ABC):
    """
    Base class for encoders, encodes and decodes matrices
    abstract methods for encoding/decoding numbers
    """

    def __init__(self):
        pass

    @abstractmethod
    def tokenize(self, value: float) -> str:
        pass

    @abstractmethod
    def detokenize(self, s: str) -> float:
        pass


class FloatTokenizer(Tokenizer):
    def __init__(
        self, precision: int = 3, mantissa_len: int = 1, max_exponent: int = 100,
    ):
        super().__init__()
        self.precision = precision
        self.mantissa_len = mantissa_len
        self.max_exponent = max_exponent
        self.base = (self.precision + 1) // self.mantissa_len
        self.max_token = 10 ** self.base
        assert (self.precision + 1) % self.mantissa_len == 0

        self.nan_token = NAN_TOKEN
        self.inf_token = INF_TOKEN
        self.neg_inf_token = NEG_INF_TOKEN

        self.symbols: List[str] = ["+", "-"]
        self.symbols.extend([f"N{i:0{self.base}d}" for i in range(self.max_token)])
        self.symbols.extend([f"E{i}" for i in range(-max_exponent, max_exponent + 1)])
        self.symbols.extend([NAN_TOKEN, INF_TOKEN, NEG_INF_TOKEN])

        self.zero_plus = " ".join(["+", *["N" + "0" * self.base] * mantissa_len, "E0"])
        self.zero_minus = " ".join(["-", *["N" + "0" * self.base] * mantissa_len, "E0"])

    def tokenize(self, val: float) -> str:
        """
        Tokenize a float number.
        """
        assert val is not None
        if val == 0.0:
            return self.zero_plus
        elif np.isnan(val):
            return self.nan_token
        elif np.isinf(val):
            return self.inf_token if val >= 0 else self.neg_inf_token
        else:
            assert val not in [-np.inf, np.inf, np.nan]

        precision = self.precision
        str_m, str_exp = f"{val:.{precision}e}".split("e")
        m1, m2 = str_m.lstrip("-").split(".")
        m: str = m1 + m2
        assert re.fullmatch(r"\d+", m) and len(m) == precision + 1, m
        expon = int(str_exp) - precision
        if expon > self.max_exponent:
            return self.inf_token if val >= 0 else self.neg_inf_token
        if expon < -self.max_exponent:
            return self.zero_plus if val >= 0 else self.zero_minus
        assert len(m) % self.base == 0
        m_digits = [m[i : i + self.base] for i in range(0, len(m), self.base)]
        assert len(m_digits) == self.mantissa_len
        sign = "+" if val >= 0 else "-"
        return " ".join([sign, *[f"N{d}" for d in m_digits], f"E{expon}"])

    def detokenize(self, s: str) -> float:
        """
        Detokenize a float number.
        """

        if s == self.inf_token:
            return np.inf
        elif s == self.neg_inf_token:
            return -np.inf
        elif s == self.nan_token:
            return np.nan

        tokens = s.split()
        if tokens[0] not in ["-", "+"]:
            raise FloatTokenError(f"Unexpected first token: {tokens[0]}")
        if tokens[-1][0] not in ["E", "N"]:
            raise FloatTokenError(f"Unexpected last token: {tokens[-1]}")

        sign = 1 if tokens[0] == "+" else -1
        mant_str = ""
        for x in tokens[1:-1]:
            mant_str += x[1:]
        mant = int(mant_str)
        exp = int(tokens[-1][1:])
        value = sign * mant * (10 ** exp)
        return float(value)


if __name__ == "__main__":

    # python -m evariste.envs.sr.tokenizer

    def test_tokenizer():

        print("===== RUNNING FLOAT TOKENIZER TESTS")

        tokenizer = FloatTokenizer(precision=3, mantissa_len=1, max_exponent=10)
        max_err = 10 ** -tokenizer.precision
        print(f"max reconstruction error: {max_err}")

        tests: List[Tuple[float, str]] = [
            (0.0, "+ N0000 E0"),
            (-np.inf, NEG_INF_TOKEN),
            (np.inf, INF_TOKEN),
            (np.nan, NAN_TOKEN),
            (1.12 * 1e-100, "+ N0000 E0"),
            (1.12 * 1e-9, "+ N0000 E0"),
            (-1.12 * 1e-9, "- N0000 E0"),
            (0.00014142356, "+ N1414 E-7"),
            (0.0014142356, "+ N1414 E-6"),
            (0.014142356, "+ N1414 E-5"),
            (0.14142356, "+ N1414 E-4"),
        ]

        x = 1.234567890123456
        for i in range(-20, 21):
            p = i - tokenizer.precision
            if p < -tokenizer.max_exponent:
                tok = "+ N0000 E0"
            elif p > tokenizer.max_exponent:
                tok = INF_TOKEN
            else:
                tok = f"+ N1235 E{p}"
            tests.append((x * (10 ** i), tok))

        for x, y in tests:

            y_ = tokenizer.tokenize(x)
            if y != y_:
                raise Exception(f'Expected "{y}" when tokenizing {x}, got "{y_}"')
            x_ = tokenizer.detokenize(y)

            # check for NaN
            if math.isnan(x) != math.isnan(x_):
                raise Exception(f'Only one is nan: {x} != "{x_}"')
            elif math.isnan(x):
                continue

            # no NaN. check for infs
            if math.isinf(x) and not math.isinf(x_):
                raise Exception(f"If x is inf, x_ should be.")
            elif math.isinf(x) and math.isinf(x_) and x != x_:
                raise Exception(f'Different infs: {x} != "{x_}"')
            elif math.isinf(x) or math.isinf(x_):
                continue

            # no infs. check close enough
            if not math.isclose(x, x_, abs_tol=max_err, rel_tol=max_err):
                raise Exception(f'Expected "{x}" when detokenizing {y}, got "{x_}"')

        print("OK")

    test_tokenizer()
