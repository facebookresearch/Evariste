# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Tuple, NamedTuple, Optional, List, Any
from evariste.model.data.dictionary import EOS_WORD, EOU_WORD, UNPROVED_WORD

FWD_INP_FMT = ("goal", "graph")
GEN_INP_FMT = ("nogoal", "graph")
LEGACY_FWD_INP_FMT = ("goal", "stack")

LABEL_SUBST_FMT = ("label", "subst")
# deprecated: please put the "proved" token in the graph
DEPRECATED_PROVED_LABEL_SUBST_FMT = ("proved", "label", "subst")
LEGACY_CMD_FMT = ("bcmd", "label", "subst", "ecmd")


class MMFwdFormat(NamedTuple):
    """
    Class to handle the format of the forward prover. Indeed the format is quite
    complicated and depends on the task and some parameters in TrainerArgs
    This class is also at training time to format the inputs, commands and auxiliary
    predictions
    """

    inp_fmt: Tuple[str, ...]
    cmd_fmt: Tuple[str, ...]
    stop_symbol: str
    cmd_prefix: Optional[List[str]]
    aux_predictions: Optional[List[str]]
    label_conditioning: bool
    proved_conditioning: bool
    reward_quantile_conditioning: bool
    use_stop_action: bool
    is_generation: bool
    hum_vs_gen_disc: bool

    @classmethod
    def from_task(cls, task: str, params: Any) -> "MMFwdFormat":
        """
        Extract the format from TrainerArgs for a given task
        """
        from evariste.trainer.args import TrainerArgs

        assert isinstance(params, TrainerArgs)
        assert is_fwd_or_gen(task)
        s = task.split("_")
        s_y = LABEL_SUBST_FMT
        hum_vs_gen_disc = False
        if task == "mm_fwd_graph2subst_seq2seq":
            s_x = LEGACY_FWD_INP_FMT  # legacy fmt
            s_y = LEGACY_CMD_FMT  # legacy fmt
        elif task in ["mm_fwd_seq2seq", "mm_fwd_rl"]:
            s_x = FWD_INP_FMT
        elif task in ["mm_gen_seq2seq", "mm_gen_rl", "mm_gen_critic_rl", "mm_gen_disc"]:
            s_x = GEN_INP_FMT
            if params.mm.cond_gen_with_proved:
                s_y = DEPRECATED_PROVED_LABEL_SUBST_FMT
            if task.endswith("_disc"):
                hum_vs_gen_disc = True
        elif s[1] == "fwd" and s[2] == "x2y" and s[-1] == "seq2seq":
            assert len(s) == 5, s
            s_x, s_y = [tuple(t.split("-")) for t in s[3].split("--")]
        else:
            raise NotImplementedError(task)

        stop_symbol = EOS_WORD
        inp_fmt, s_y = s_x, s_y
        aux_predictions = None
        if "EOU" in s_y:
            cmd_fmt = s_y[: s_y.index("EOU")]
            aux_predictions = list(s_y[s_y.index("EOU") + 1 :])
            stop_symbol = EOU_WORD
        else:
            cmd_fmt = s_y

        if params.mm.cond_gen_with_proved:
            # not needed anymore if putting the "proved" token in encoder.
            # keeping it for bwd compatibility for the moment
            assert "proved" in s_y, s_y
            prefix = [UNPROVED_WORD]
        else:
            prefix = None

        _is_gen = is_gen_task(task)
        label_conditioning = params.mm.graph.label_conditioning
        if label_conditioning:
            assert _is_gen, "label_conditioning only used in generation"
        proved_conditioning = params.mm.graph.proved_conditioning
        if proved_conditioning:
            assert _is_gen, "proved_conditioning only used in generation"
        reward_quantile_conditioning = params.mm.graph.reward_quantile_conditioning
        if reward_quantile_conditioning:
            assert _is_gen, "reward_quantile_conditioning only used in generation"
            assert (
                params.rl_params.replay_buffer.n_reward_quantiles > 0
            ), "needed for reward_quantile_conditioning"

        return cls(
            inp_fmt=inp_fmt,
            cmd_fmt=cmd_fmt,
            stop_symbol=stop_symbol,
            cmd_prefix=prefix,
            aux_predictions=aux_predictions,
            label_conditioning=label_conditioning,
            proved_conditioning=proved_conditioning,
            reward_quantile_conditioning=reward_quantile_conditioning,
            use_stop_action=params.mm.stop_action and _is_gen,
            is_generation=_is_gen,
            hum_vs_gen_disc=hum_vs_gen_disc,
        )

    @classmethod
    def from_trainer_args(cls, params: Any) -> "MMFwdFormat":
        """
        When creating a ForwardProver from TrainerArgs, we need to know the format of
        the inputs, commands of the ForwardProver. With this helper we extract this fmts
        from TrainerArgs + we check that the
        format defined for each fwd/gen task in TrainerArgs are compatible (
        sanity check to avoid weird bugs if fmts are different between tasks)
        """
        from evariste.trainer.args import TrainerArgs

        assert isinstance(params, TrainerArgs)
        fwd_tasks = [t for t in params.parsed_tasks("mm") if is_fwd_or_gen(t)]

        if len(fwd_tasks) == 0:
            raise ValueError

        base_task, *other_tasks = fwd_tasks
        base_format = cls.from_task(base_task, params=params)
        for other_task in other_tasks:
            other_format = cls.from_task(base_task, params=params)
            if other_format != base_format:
                raise ValueError(
                    f"Found different formats for task "
                    f"{base_task} and {other_task} "
                    f"({base_format} != {other_format})"
                )
        return base_format


def is_fwd_task(task: str) -> bool:
    return task.startswith("mm_fwd")


def is_gen_task(task: str) -> bool:
    return task.startswith("mm_gen")


def is_fwd_or_gen(task: str) -> bool:
    return is_fwd_task(task) or is_gen_task(task)


def use_graph_sampler(task: str) -> bool:
    return is_fwd_or_gen(task) and (task.endswith("_seq2seq") or task.endswith("_disc"))
    # allows to create the necessary data to sample graphs


def use_critic(task: str) -> bool:
    assert is_fwd_or_gen(task), task
    return "critic" in task
