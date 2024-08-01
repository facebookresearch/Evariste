# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import pytest

# from evariste.model.data.dictionary import (
#     B_CMD_WORD,
#     E_CMD_WORD,
#     B_HYP_WORD,
#     M_HYP_WORD,
#     E_HYP_WORD,
#     B_GOAL_WORD,
#     E_GOAL_WORD,
# )

# from evariste.backward.env.hol_light import (
#     HLVecBackwardEnv,
#     HLTheorem,
#     HLTactic,
# )
# from evariste.backward.env.hol_light.env import HLBackwardEnv

# TODO: fix
# @pytest.mark.slow
# class TestHLBackwardEnv:
#     single_env_instance = None
#
#     def _get_env(self) -> HLBackwardEnv:
#         if type(self).single_env_instance is None:
#             type(self).single_env_instance = HLBackwardEnv()
#         return type(self).single_env_instance
#
#     def test_gen_cmd_root_goal(self):
#         th = HLTheorem(
#             [B_GOAL_WORD]
#             + ("! m n p . m * ( n DIV p ) <= ( m * n ) DIV p").split()
#             + [E_GOAL_WORD]
#         )
#         tact = HLTactic([B_CMD_WORD] + "e ( REPEAT GEN_TAC ) ;;".split() + [E_CMD_WORD])
#         cmd = HLBackwardEnv.gen_cmd(theorem=th, tactic=tact)
#         expected_cmd = [
#             "g `! m n p . m * ( n DIV p ) <= ( m * n ) DIV p` ;;",
#             "e ( REPEAT GEN_TAC ) ;;",
#         ]
#
#         assert cmd == expected_cmd
#
#     def test_gen_cmd_subgoal_no_state(self):
#         th = HLTheorem(
#             [B_GOAL_WORD, B_HYP_WORD, M_HYP_WORD]
#             + "~ ( p = 0 )".split()
#             + [E_HYP_WORD]
#             + "m * n DIV p <= ( m * n ) DIV p".split()
#             + [E_GOAL_WORD]
#         )
#         tact = HLTactic([B_CMD_WORD] + "e ( REPEAT GEN_TAC ) ;;".split() + [E_CMD_WORD])
#         with pytest.raises(CmdToEnvError):
#             HLBackwardEnv.gen_cmd(theorem=th, tactic=tact)
#
#     def test_gen_cmd_subgoal(self):
#         th = HLTheorem(
#             tokens=[B_GOAL_WORD, B_HYP_WORD, M_HYP_WORD]
#             + "~ ( p = 0 )".split()
#             + [E_HYP_WORD]
#             + "m * n DIV p <= ( m * n ) DIV p".split()
#             + [E_GOAL_WORD],
#             state="dummystate",
#         )
#         tact = HLTactic([B_CMD_WORD] + "e ( REPEAT GEN_TAC ) ;;".split() + [E_CMD_WORD])
#         cmd = HLBackwardEnv.gen_cmd(theorem=th, tactic=tact)
#         expected_cmd = [
#             f"restore_goal dummystate ;;",
#             "p() ;;",
#             "e ( REPEAT GEN_TAC ) ;;",
#         ]
#         assert cmd == expected_cmd
#
#     def test_execute_on_root_goal(self):
#         hl_env = self._get_env()
#         th = HLTheorem(
#             [B_GOAL_WORD] + "m * n DIV p <= ( m * n ) DIV p".split() + [E_GOAL_WORD]
#         )
#         tact = HLTactic(
#             [B_CMD_WORD] + "e ( ASM_CASES_TAC ` p = 0 ` ) ;;".split() + [E_CMD_WORD]
#         )
#         sub_th_1 = HLTheorem(
#             tokens=[B_GOAL_WORD, B_HYP_WORD, M_HYP_WORD]
#             + "p = 0".split()
#             + [E_HYP_WORD]
#             + "m * n DIV p <= ( m * n ) DIV p".split()
#             + [E_GOAL_WORD]
#         )
#         sub_th_2 = HLTheorem(
#             tokens=[B_GOAL_WORD, B_HYP_WORD, M_HYP_WORD]
#             + "~ ( p = 0 )".split()
#             + [E_HYP_WORD]
#             + "m * n DIV p <= ( m * n ) DIV p".split()
#             + [E_GOAL_WORD]
#         )
#         expected_out_theorems = {sub_th_1, sub_th_2}
#         cmd = HLBackwardEnv.gen_cmd(theorem=th, tactic=tact)
#         output = hl_env.execute(cmd)
#         out_theorems = HLBackwardEnv.parse_to_theorems(output)
#         assert set(out_theorems) == expected_out_theorems
#
#     def test_execute_on_subgoal(self):
#         hl_env = self._get_env()
#         th = HLTheorem(
#             tokens=[B_GOAL_WORD, B_HYP_WORD, M_HYP_WORD]
#             + "~ ( p = 0 )".split()
#             + [E_HYP_WORD]
#             + "m * n DIV p <= ( m * n ) DIV p".split()
#             + [E_GOAL_WORD],
#             state='"\\132\\149\\166\\190\\000\\000\\001\\b\\000\\000\\000r\\000\\000\\001>\\000\\000\\001:\\160\\160\\160 \\144\\160\\160\\162\\161!~\\161#fun\\160\\161$bool@\\160\\161$bool@@\\162\\162\\161!=\\161#fun\\160\\161#num@\\160\\161#fun\\160\\004\\006\\160\\161\\004\\016@@@\\160!p\\161\\004\\n@\\162\\161\'NUMERAL\\161#fun\\160\\161#num@\\160\\004\\003@\\161\\"_0\\004\\005@\\004(@\\162\\162\\161\\"<=\\161#fun\\160\\161\\004\\029@\\160\\161\\004\\005\\160\\161\\004!@\\160\\161$bool@@@\\162\\162\\161!*\\161\\004\\015\\160\\161\\004+@\\160\\161\\004\\019\\160\\161\\004\\030@\\160\\161\\0041@@@\\160!m\\161\\0044@\\162\\162\\161#DIV\\161#fun\\160\\161\\004+@\\160\\161\\004\\005\\160\\161\\004@@\\160\\161\\004B@@@\\160!n\\161\\0044@\\160!p\\161\\004H@\\162\\162\\161#DIV\\004\\020\\162\\162\\161\\004(\\004\'\\160!m\\161\\004R@\\160!n\\161\\004D@\\160!p\\161\\004X@"',
#         )
#         tact = HLTactic(
#             [B_CMD_WORD] + "e ( ASM_SIMP_TAC [ LE_RDIV_EQ ] ) ;;".split() + [E_CMD_WORD]
#         )
#         sub_th = HLTheorem(
#             tokens=[B_GOAL_WORD, B_HYP_WORD, M_HYP_WORD]
#             + "~ ( p = 0 )".split()
#             + [E_HYP_WORD]
#             + "p * m * n DIV p <= m * n".split()
#             + [E_GOAL_WORD]
#         )
#         expected_out_theorems = {sub_th}
#         cmd = HLBackwardEnv.gen_cmd(theorem=th, tactic=tact)
#         output = hl_env.execute(cmd)
#         out_theorems = HLBackwardEnv.parse_to_theorems(output)
#         assert set(out_theorems) == expected_out_theorems
#
#     def test_execute_fails(self):
#         hl_env = self._get_env()
#         with pytest.raises(FailedEnvError):
#             hl_env.execute(0)
#
#
# @pytest.mark.slow
# class TestHLVecBackwardEnv:
#     def test_vec_single(self):
#         envs = HLVecBackwardEnv(num_envs=1)
#         theorem = HLTheorem(
#             HLTheorem.tokenize(
#                 hyps=[], conclusion="! m n p . m * ( n DIV p ) <= ( m * n ) DIV p"
#             )
#         )
#         tact = HLTactic(HLTactic.tokenize(cmd="ASM_CASES_TAC ` p = 0 `"))
#         envs.apply_tactics([theorem], [[tact]])
#
#     def test_vec_multi(self):
#         envs = HLVecBackwardEnv(num_envs=10)
#         theorem = HLTheorem(
#             HLTheorem.tokenize(
#                 hyps=[], conclusion="! m n p . m * ( n DIV p ) <= ( m * n ) DIV p"
#             )
#         )
#         tact = HLTactic(HLTactic.tokenize(cmd="ASM_CASES_TAC ` p = 0 `"))
#         envs.apply_tactics([theorem] * 2, [[tact] * 3] * 2)

#
# if __name__ == "__main__":
#     t = TestHLVecBackwardEnv()
#     t.test_vec_multi()
