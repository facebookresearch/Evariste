# Copyright (c) Facebook, Inc. and its affiliates.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Dict, Union
from dataclasses import dataclass
import random
import string
import re


def parse_goal(*args, **kwargs):
    pass  # TODO delete and stop importing


@dataclass
class Hypothesis:
    hyp: str
    n_hyps: int

    def to_expression(self) -> str:
        to_add_out = ""
        # hypothesis
        if self.hyp[0] != " ":
            to_add_out += "∀ "
            for c in self.hyp:
                if c == ":":
                    break
        to_add_out += self.hyp.strip()
        if to_add_out[-1] != ",":
            to_add_out += ","
        to_add_out += "\n"
        # to_add_out contains the whole block for this hypotheses.
        # If a := is in there, replace forall with let and last "," with " in"
        if ":=" in to_add_out:
            to_add_out = to_add_out.replace("∀ ", "let ")
            to_add_out = to_add_out.replace(",\n", " in\n")
        return to_add_out

    def to_command(self) -> str:
        sanitized = self.hyp.strip(" ,\n")
        return f"({sanitized})"


@dataclass
class StructuredGoal:
    hyps: List[Hypothesis]
    conclusion: str

    def to_expression(self) -> Dict[str, Union[str, int]]:
        cur_out = ""
        n_inst = 0
        for hyp in self.hyps:
            to_add = hyp.to_expression()
            cur_out += to_add
            instance_parameters = {": decidable", ": nonempty", ": setoid", ": fintype"}
            if to_add.startswith("∀ _inst") or any(
                ip in to_add for ip in instance_parameters
            ):
                cur_out += "by {tactic.unfreeze_local_instances, tactic.freeze_local_instances, exact\n"
                n_inst += 1
        cur_out += self.conclusion
        cur_out += "\n}" * n_inst

        replaces = [
            ("Sort ?", "Sort*"),
            ("Type ?", "Type*"),
            ("⇑", ""),
            ("↥", ""),
        ]
        for before, after in replaces:
            cur_out = cur_out.replace(before, after)

        return {"goal": cur_out, "n_hyps": sum(h.n_hyps for h in self.hyps)}

    def to_command(self) -> Dict[str, Union[str, int]]:
        command = (
            " ".join(h.to_command() for h in self.hyps)
            + " : "
            + self.conclusion.replace("\n", " ")
        )
        rstr = "".join(random.choice(string.ascii_lowercase) for _ in range(6))
        return {
            "command": f"def _root_.{rstr} {command} := sorry",
            "decl_name": rstr,
            "n_hyps": sum(h.n_hyps for h in self.hyps),
        }


def parse_goal_structured(goal_pp: str) -> List[StructuredGoal]:

    # remove lines starting with case
    goal_pp = "\n".join(
        line
        for line in goal_pp.split("\n")
        if not (line.startswith("case ") and " :" not in line)
    )

    # re-merge lines split by goal_pp line wrap. merge two lines if the above one
    # does not end with a "," and that the below one does not start with a "⊢".
    # also remove extra spaces
    goal_pp = re.sub(r"(?<![,\n])\n *(?![⊢\n])", r" ", goal_pp)

    res, line_id = [], 0
    lines = goal_pp.split("\n")
    while line_id < len(lines):
        these_hyps = []
        this_conc = ""
        while line_id < len(lines):
            line = lines[line_id]
            if len(line) == 0:
                line_id += 1
                continue
            if re.match("^[0-9]+ goals$", line):
                line_id += 1
                continue
            if line.startswith("⊢"):
                this_conc = line[1:].strip()
                line_id += 1
                while line_id < len(lines):
                    if len(lines[line_id].strip()) == 0:
                        line_id += 1
                        break
                    this_conc += "\n" + lines[line_id]
                    line_id += 1
                break
            else:
                local_hyps = 0
                # hypothesis
                if line[0] != " ":
                    for c in line:
                        if c == " ":
                            local_hyps += 1
                        if c == ":":
                            break
                this_hyp = line.strip()
                line_id += 1
                while line_id < len(lines) and (
                    len(lines[line_id]) == 0 or lines[line_id][0] == " "
                ):
                    this_hyp += " " + lines[line_id]
                    line_id += 1
                these_hyps.append(Hypothesis(hyp=this_hyp, n_hyps=local_hyps))

        res.append(StructuredGoal(hyps=these_hyps, conclusion=this_conc))
    return res
