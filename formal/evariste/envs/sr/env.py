# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from datetime import datetime
import sys
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, field
import math
import warnings
import numpy as np
import numexpr as ne

from evariste.utils import MyTimeoutError
from params import Params, ConfStore
from evariste.envs.eq.rules import TRule
from evariste.envs.eq.graph import Node, ZERO, infix_for_numexpr  # , INode, VNode
from evariste.envs.eq.env import EquationsEnvArgs, EquationEnv
from evariste.envs.sr.rules import RULES, INIT_RULE
from evariste.envs.sr.simplification import simplify_via_rules


@dataclass
class Transition:
    # applying the rule in forward on eq gives next_eq
    eq: Node
    rule: TRule
    prefix_pos: int
    tgt_vars: Dict[str, Node]
    next_eq: Node

    def __post_init__(self):
        assert self.tgt_vars.keys() == self.rule.r_vars - self.rule.l_vars
        applied = self.rule.apply(self.eq, fwd=True, prefix_pos=self.prefix_pos)
        next_eq = applied["eq"].set_vars(self.tgt_vars)
        assert next_eq.eq(self.next_eq)


@dataclass
class XYValues:
    x: List[float]
    y: List[float]

    def __post_init__(self):
        assert isinstance(self.x, list), "Found x: {}".format(self.x)
        assert isinstance(self.y, list), "Found y: {}".format(self.y)
        assert not any(math.isnan(x) for x in self.x)
        assert len(self.x) == len(self.y) > 0, (len(self.x), len(self.y))

    def __len__(self) -> int:
        return len(self.x)


@dataclass
class SREnvArgs(Params):
    eq_env: EquationsEnvArgs
    until_depth: int = field(
        default=0,
        metadata={
            "help": "Final desired depth of expressions during backtrack generation."
        },
    )
    max_backward_steps: int = field(
        default=1000,
        metadata={
            "help": (
                "Maximum number of backward steps during backtract generation. "
                "Is generally used with until_depth = 0 to limit sequence length."
            )
        },
    )
    # rule_criterion: Callable[[Node], bool] = field(
    #     default=lambda x: True,
    #     metadata={"help": "Whether to keep the rule backtrack or not."},
    # )
    sampling_strategy: str = field(
        default="intervals", metadata={"help": "How are x datapoints generated?"},
    )
    min_n_points: int = field(
        default=1, metadata={"help": "How many x datapoints are generated?"},
    )
    max_n_points: int = field(
        default=200, metadata={"help": "How many x datapoints are generated?"},
    )
    rtol: float = field(
        default=1e-5,
        metadata={"help": "Relative tolerance for values to be considered equal."},
    )
    atol: float = field(
        default=1e-8,
        metadata={"help": "Absolute tolerance for values to be considered equal."},
    )

    def __post_init__(self):
        assert self.max_n_points >= self.min_n_points and self.min_n_points >= 1
        assert self.sampling_strategy in ["intervals", "uniform"]


class SREnv:
    def __init__(
        self,
        eq_env: EquationEnv,
        rules: List[TRule],
        until_depth: int,
        max_backward_steps: int,
        # rule_criterion: Callable[[Node], bool] = lambda x: True,
        sampling_strategy: str,
        min_n_points: int,
        max_n_points: int,
        rtol: float,
        atol: float,
    ):
        self.eq_env = eq_env
        self.rules = rules
        self.until_depth = until_depth
        # self.rule_criterion = rule_criterion
        self.max_backward_steps = max_backward_steps
        self.sampling_strategy = sampling_strategy
        self.max_n_points = max_n_points
        self.min_n_points = min_n_points
        self.rtol = rtol
        self.atol = atol

    @property
    def rng(self):
        return self.eq_env.rng

    def set_rng(self, rng):
        old_rng = self.eq_env.set_rng(rng)
        return old_rng

    @staticmethod
    def build(args: SREnvArgs, seed: Optional[int] = None):
        eq_env = EquationEnv.build(args=args.eq_env, seed=seed)
        return SREnv(
            eq_env=eq_env,
            rules=RULES,
            until_depth=args.until_depth,
            max_backward_steps=args.max_backward_steps,
            # rule_criterion=args.rule_criterion,
            sampling_strategy=args.sampling_strategy,
            min_n_points=args.min_n_points,
            max_n_points=args.max_n_points,
            rtol=args.rtol,
            atol=args.atol,
        )

    def is_identical(self, curr_xy: XYValues, true_xy: XYValues) -> bool:
        """
        Determine whether sets of values are identical.
        """
        assert curr_xy.x == true_xy.x
        res = np.isclose(
            curr_xy.y, true_xy.y, rtol=self.rtol, atol=self.atol, equal_nan=True
        )
        return res.all().item()

    def eligible_actions(self, eq: Node) -> List[Tuple[TRule, int]]:
        """
        Return the list of rules that can be applied to the current equation,
        and in which prefix position.
        """
        results: List[Tuple[TRule, int]] = []
        for rule in self.rules:
            if rule.rule_type == "special":
                continue
            res: List[Tuple[int, Node]] = rule.eligible(eq, fwd=False)
            for prefix_pos, eq_match in res:
                # e.g. verify that the matched node is not too deep
                # if self.rule_criterion(eq_match):  # TODO: fix
                results.append((rule, prefix_pos))
        return results

    def sample_transition(self, eq: Node) -> Optional[Transition]:
        actions: List[Tuple[TRule, int]] = self.eligible_actions(eq)
        if len(actions) == 0:
            return None
        action_id = self.rng.choice(len(actions))
        rule, prefix_pos = actions[action_id]
        applied = rule.apply(eq, fwd=False, prefix_pos=prefix_pos)
        tgt_vars = {k: v for k, v in applied["match"].items() if k not in rule.l_vars}

        return Transition(
            eq=applied["eq"],
            rule=rule,
            prefix_pos=prefix_pos,
            tgt_vars=tgt_vars,
            next_eq=eq,
        )

    def sample_trajectory(self, final_eq: Node) -> List[Transition]:

        trajectory: List[Transition] = []
        curr_eq = final_eq
        step = 0

        # start from the final equation and reduce it
        while curr_eq.depth() >= self.until_depth and step < self.max_backward_steps:
            transition = self.sample_transition(curr_eq)
            if transition is None:
                break
            curr_eq = transition.eq
            trajectory.append(transition)
            step += 1

        # initial transition
        ##TODO: not sure whethere this is relevant in the case we don't backtrack until the ZERO expr.
        init_transition = Transition(
            eq=ZERO,
            rule=INIT_RULE,
            prefix_pos=0,
            tgt_vars={"A": curr_eq},
            next_eq=curr_eq,
        )
        trajectory.append(init_transition)

        return trajectory[::-1]

    def check_trajectory(self, target_eq: Node, trajectory: List[Transition]):
        assert trajectory[0].eq.eq(ZERO)
        assert trajectory[-1].next_eq.eq(target_eq)
        for a, b in zip(trajectory[:-1], trajectory[1:]):
            assert a.next_eq.eq(b.eq)
            assert len(trajectory) <= self.max_backward_steps

    @staticmethod
    def print_trajectory(trajectory: List[Transition]) -> None:
        for step in trajectory:
            assert isinstance(step, Transition)
            print(f"{step.eq.infix():<80} -> {step.next_eq.infix()}")

    # @staticmethod
    # @timeout  # TODO add timeout and try / except
    # def evaluate_at(eq: Node, x: List[float]) -> List[float]:
    #    eq_vars = eq.get_vars()
    #    assert (len(eq_vars) == 1 and "x0" in eq_vars) or len(
    #        eq_vars
    #    ) == 0, "Eq: {}, eq_vars: {}".format(eq, eq_vars)
    #    try:
    #        y = [eq._evaluate(subst={"x0": v}, none_if_nan_inf=False) for v in x]
    #    except MyTimeoutError:  # TODO: improve this
    #        y = [math.nan for _ in x]
    #    return y
    @staticmethod
    def compute_numexpr(eq: Node, x: Dict[str, np.ndarray]) -> List[float]:
        keys = list(x.keys())
        assert len(keys) > 0, "No vars found"

        infix = infix_for_numexpr(eq.infix())
        _len = x[keys[0]].shape[0]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            try:
                values = ne.evaluate(infix, local_dict=x).tolist()
                for i, value in enumerate(values):
                    if isinstance(value, complex):
                        values[i] = math.nan
            except (
                ZeroDivisionError,
                OverflowError,
                AttributeError,
                TypeError,
                ValueError,
                KeyError,
                MemoryError,
            ):
                values = [math.nan] * _len

            return values

    @staticmethod
    def evaluate_at(eq: Node, x: List[float]) -> List[float]:

        if not eq.can_evaluate():
            now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            print(
                f"WARNING {now} -- cannot evaluate expression {eq.infix()}",
                file=sys.stderr,
                flush=True,
            )
            return [math.nan] * len(x)

        if not eq.has_vars():
            value = eq._evaluate(subst=None, none_if_nan_inf=False)
            if isinstance(value, complex):
                value = math.nan
            return [value] * len(x)

        eq_vars = eq.get_vars()
        assert len(eq_vars) == 1 and "x0" in eq_vars, "Eq: {}, eq_vars: {}".format(
            eq, eq_vars
        )
        y = SREnv.compute_numexpr(eq, {"x0": np.array(x)})
        return y

    def sample_dataset(self, eq: Node):
        n_points = self.rng.randint(self.min_n_points, self.max_n_points + 1)
        if self.sampling_strategy == "intervals":
            x_arr = np.linspace(start=-10, stop=10, num=n_points)
        elif self.sampling_strategy == "uniform":
            x_arr = self.rng.uniform(-10, 10, size=(n_points,))
        else:
            raise NotImplementedError
        x: List[float] = x_arr.tolist()
        y = self.evaluate_at(eq, x)
        return XYValues(x=x, y=y)


if __name__ == "__main__":

    # python -m evariste.envs.sr.env
    # python -m evariste.envs.sr.tokenizer
    # python -m evariste.envs.sr.generation

    import evariste.datasets

    sr_args = SREnvArgs(
        eq_env=ConfStore["sr_eq_env_default"], max_backward_steps=100, max_n_points=10
    )
    env = SREnv.build(sr_args, seed=None)

    def test_trajectory_generation(n_tests: int):

        print("===== TEST TRAJECTORY GENERATION")

        # x = VNode("x0")
        # y = VNode("x1")
        # c = INode(4)
        # expr = (((x * c).cos() + 2).exp() + 4 * x + 1) * 2 + y)

        for _ in range(n_tests):
            expr = env.eq_env.generate_expr(n_ops=10)
            print(f"Target: {expr}")
            trajectory = env.sample_trajectory(expr)
            env.check_trajectory(expr, trajectory)
            env.print_trajectory(trajectory)
            print("")

        print("OK")

    test_trajectory_generation(n_tests=50)
