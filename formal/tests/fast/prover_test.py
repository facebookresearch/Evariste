# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import unittest
from unittest.mock import patch, Mock, call
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from contextlib import closing
from evariste.backward.env.core import EnvExpansion
from evariste.backward.env.lean.env import DeclNotFound, LeanInstanceDied
from evariste.backward.prover.core import ProofResult

from evariste.backward.prover.prover import (
    BatchBackwardRunner,
    ProofRecord,
    ProofStatus,
)
from evariste.backward.prover.prover_args import ProverParams
from evariste.backward.graph import BackwardGoal, Theorem, UnMaterializedTheorem
from evariste.backward.env.metamath.graph import MMTheorem
from evariste.async_workers.async_worker_helpers import make_iterator


test_dump_path = Path("/dump_path/")
test_beam_path = Path("/beam_path/")

test_prover_params = ProverParams(
    mcts=Mock(),
    beam_path=test_beam_path,
    dump_path=test_dump_path,
    n_simultaneous_proofs=10,
    beam_kind=Mock(),
)


class MockEnvGen(Mock):
    pass


class NoInheritMock(Mock):
    def _get_child_mock(self, **kw):
        return Mock(**kw)


class MockWithMethod(NoInheritMock):
    def __init__(self, n_ret: List[int]):
        super().__init__()
        self.n_ret = n_ret

    def ready_env_expansions(self, prover_params):
        return [Mock() for _ in range(self.n_ret.pop(0))]

    def ready_model_expansions(self):
        return [Mock() for _ in range(self.n_ret.pop(0))]

    def get_theorems_to_expand(self):
        return [Mock() for _ in range(self.n_ret.pop(0))]

    def materialize_goal(self, g: BackwardGoal) -> BackwardGoal:
        assert isinstance(g.theorem, UnMaterializedTheorem)
        g.theorem.batch_id = "being_materialized"
        return g


class MockHandler(NoInheritMock):
    def __init__(
        self,
        goal: BackwardGoal,
        to_expand: Optional[List[Optional[List[Theorem]]]] = None,
        res: Optional[ProofResult] = None,
    ):
        super().__init__()
        self.goal = goal
        self.to_expand = to_expand
        self.done = False
        self.to_send = 0
        self.res = res

    def get_theorems_to_expand(self):
        assert isinstance(self.to_expand, list)
        self.to_send += 1
        return self.to_expand.pop(0)

    def get_done(self) -> bool:
        assert self.to_expand is not None
        if (
            len(self.to_expand) > 0
            or len(self.send_env_expansions.mock_calls) < self.to_send
        ):
            return True
        self.done = True
        return True

    def get_result(self):
        return self.res


class MockExpansionHandler(NoInheritMock):
    def __init__(
        self,
        ready: Optional[List[Optional[List[Tuple[int, EnvExpansion]]]]] = None,
        pre_ready: Optional[Dict[int, List[EnvExpansion]]] = None,
    ):
        super().__init__()
        self.ready = ready
        self.pre_ready = pre_ready
        self.env.has_died = set()
        self.next_to_send: List[Tuple[int, EnvExpansion]] = []

    def ready_env_expansions(self, prover_params: ProverParams):
        if self.ready is not None:
            return self.ready.pop(0)
        ret = self.next_to_send
        self.next_to_send = []
        return ret

    def send_theorems_to_expand(
        self, proof_id: int, to_expand: List[Theorem], params=None
    ):
        assert self.pre_ready is not None
        self.next_to_send.append((proof_id, self.pre_ready[proof_id].pop(0)))


class MockBackwardEnv(NoInheritMock):
    def __init__(self):
        super().__init__()
        self.expander_env.theorem_ready = lambda x: MMTheorem(conclusion=x, hyps=[])

    def materialize_goal(self, goal: BackwardGoal):
        assert isinstance(goal.theorem, UnMaterializedTheorem)
        goal.theorem.batch_id = goal.theorem.conclusion
        return goal


@patch(
    "evariste.backward.prover.prover.backward_init",
    return_value=(Mock(), Mock(), Mock()),
)
@patch("evariste.backward.prover.prover.get_prover_handler", return_value=(Mock()))
@patch("evariste.backward.prover.prover.os.makedirs")
@patch("evariste.backward.prover.prover.GPUMonitor")
@patch("evariste.backward.prover.prover.Logger")
@patch("evariste.backward.prover.prover.get_tail_logger")
@patch("evariste.backward.prover.prover.json.dumps")
class MockingTestTestCase(unittest.TestCase):
    def test_create_close(
        self,
        mocked_json_dumps,
        mocked_tail_logger,
        mocked_logger,
        mocked_gpumonitor,
        mocked_makedirs,
        mocked_handler,
        mocked_backward_init,
    ):
        bbr_logs = test_dump_path / "runner_logs" / "batch_backward_runner"
        bbr = BatchBackwardRunner(Mock(), Mock(), test_prover_params, "decoder_type")
        mocked_makedirs.assert_has_calls([call(bbr_logs, exist_ok=True)])
        mocked_logger.assert_has_calls(
            [
                call(
                    outdir=bbr_logs,
                    tag="ProofHandler",
                    quiet=test_prover_params.quiet,
                    only_jsonl=test_prover_params.only_jsonl,
                ),
                call(
                    outdir=bbr_logs,
                    tag="env",
                    quiet=test_prover_params.quiet,
                    only_jsonl=test_prover_params.only_jsonl,
                ),
            ]
        )

        mocked_tail_logger.assert_has_calls(
            [
                call(
                    test_dump_path / "mcts_logs" / "batch_backward_runner.log",
                    loc_info=True,
                )
            ]
        )
        bbr.close()

    def test_create_fail(
        self,
        mocked_json_dumps,
        mocked_tail_logger,
        mocked_logger,
        mocked_gpumonitor,
        mocked_makedirs,
        mocked_handler,
        mocked_backward_init,
    ):
        """ This checks that an error in bbr doesn't lead to deadlock with a hanging gpu_mon"""
        for mocked_obj in [
            mocked_tail_logger,
            mocked_logger,
            mocked_makedirs,
            mocked_handler,
            mocked_backward_init,
        ]:
            mocked_obj.side_effect = Exception
            with self.assertRaises(Exception):
                BatchBackwardRunner(Mock(), Mock(), test_prover_params, "decoder_type")
            mocked_obj.side_effect = None

    def test_alive(
        self,
        mocked_json_dumps,
        mocked_tail_logger,
        mocked_logger,
        mocked_gpumonitor,
        mocked_makedirs,
        mocked_handler,
        mocked_backward_init,
    ):
        mocked_backward_init.return_value = (
            Mock(),
            MockWithMethod([0]),
            MockWithMethod([0]),
        )

        bbr = BatchBackwardRunner(Mock(), Mock(), test_prover_params, "decoder_type")
        assert not bbr.is_alive()
        bbr.start()
        assert bbr.is_alive()
        assert bbr.ready() == []
        bbr.stop()
        assert not bbr.is_alive()
        with self.assertRaises(AssertionError):
            bbr.ready()
        bbr.close()

    def test_grab(
        self,
        mocked_json_dumps,
        mocked_tail_logger,
        mocked_logger,
        mocked_gpumonitor,
        mocked_makedirs,
        mocked_handler,
        mocked_backward_init,
    ):
        mocked_handler.return_value = MockWithMethod([0])
        mocked_backward_init.return_value = (
            Mock(),
            MockWithMethod([0]),
            MockWithMethod([0]),
        )
        bbr = BatchBackwardRunner(Mock(), Mock(), test_prover_params, "decoder_type")

        bbr.start()
        goal = BackwardGoal(theorem=UnMaterializedTheorem("x"))
        assert bbr.grab_proofs() == []
        rid = bbr.submit(goal)
        assert rid == 0
        bbr.grab_proofs()
        assert len(bbr.cur_proofs) == 1 and 0 in bbr.cur_proofs

        bbr.env.materialize_goal = Mock(side_effect=DeclNotFound)
        rid = bbr.submit(goal)
        grabbed = bbr.grab_proofs()
        assert (grabbed[0][0], grabbed[0][1][1]) == (1, {})
        bbr.close()

    def test_unmat(
        self,
        mocked_json_dumps,
        mocked_tail_logger,
        mocked_logger,
        mocked_gpumonitor,
        mocked_makedirs,
        mocked_handler,
        mocked_backward_init,
    ):
        mocked_handler.return_value = lambda x: MockHandler(goal=x)
        mocked_backward_init.return_value = (
            Mock(),
            MockWithMethod([0]),
            MockWithMethod([0]),
        )
        bbr = BatchBackwardRunner(Mock(), Mock(), test_prover_params, "decoder_type")
        bbr.start()
        rid = bbr.submit(BackwardGoal(theorem=UnMaterializedTheorem(label="to_mat")))
        assert bbr.grab_proofs() == []
        g = bbr.cur_proofs[0].proof_handler.goal
        assert isinstance(g.theorem, UnMaterializedTheorem)
        assert g.theorem.batch_id == "being_materialized"
        assert g.name.startswith("to_mat")

        ph = bbr.cur_proofs[0].proof_handler
        assert isinstance(ph, Mock)
        to_ret = [None, MMTheorem(conclusion="yay materialized", hyps=[])]
        bbr.env.expander_env.theorem_ready = lambda x: to_ret.pop(0)  # type: ignore
        # first : nothing happens, on second call, theorem has been materialized, on third call, nothing happens
        bbr.check_unmaterialized()
        assert len(ph.mock_calls) == 0, ph.mock_calls
        bbr.check_unmaterialized()
        ph.send_materialized.assert_called()
        assert (
            bbr.cur_proofs[0].proof_handler.goal.theorem.conclusion
            == "yay materialized"
        )
        bbr.check_unmaterialized()
        ph.send_materialized.assert_called_once()
        bbr.close()

        # test declnotfound on materializing
        rid = bbr.submit(BackwardGoal(theorem=UnMaterializedTheorem(label="to_mat")))
        assert bbr.grab_proofs() == []

        def raise_not_found(*args):
            raise DeclNotFound

        bbr.env.expander_env.theorem_ready = raise_not_found  # type: ignore
        res = bbr.check_unmaterialized()
        assert len(res) == 1, res
        assert (
            res[0][0] == rid
            and res[0][1][0].proof == None
            and isinstance(res[0][1][0].exception, DeclNotFound)
        )

    def test_add_expander(
        self,
        mocked_json_dumps,
        mocked_tail_logger,
        mocked_logger,
        mocked_gpumonitor,
        mocked_makedirs,
        mocked_handler,
        mocked_backward_init,
    ):
        goal = BackwardGoal(theorem=UnMaterializedTheorem(label="to_mat"))
        mocked_backward_init.return_value = (Mock(), MockWithMethod([0]), Mock())
        bbr = BatchBackwardRunner(Mock(), Mock(), test_prover_params, "decoder_type")
        expected_th: List[Theorem] = [MMTheorem(conclusion="0", hyps=[])]
        bbr.cur_proofs = {
            0: ProofRecord(MockHandler(goal=goal, to_expand=[expected_th])),
            1: ProofRecord(MockHandler(goal=goal, to_expand=[[]])),
            2: ProofRecord(MockHandler(goal=goal, to_expand=[None, None])),
        }
        bbr.add_to_expander()

        assert bbr.cur_proofs[0].s.status == ProofStatus.WAITING_FOR_EXPAND
        assert bbr.cur_proofs[1].s.status == ProofStatus.WAITING_FOR_EXPAND
        assert bbr.cur_proofs[2].s.status == ProofStatus.TO_EXPAND

        assert bbr.expansion_handler.processing == {0: expected_th, 1: []}
        bbr.expander.process_async.assert_called_with(0, expected_th, None)
        assert bbr.expansion_handler.received[2] == {}

        bbr.add_to_expander()
        bbr.expander.process_async.assert_called_once()
        bbr.close()

    def test_send_env_expansions(
        self,
        mocked_json_dumps,
        mocked_tail_logger,
        mocked_logger,
        mocked_gpumonitor,
        mocked_makedirs,
        mocked_handler,
        mocked_backward_init,
    ):
        goal = BackwardGoal(theorem=UnMaterializedTheorem(label="to_mat"))
        mocked_backward_init.return_value = (Mock(), MockWithMethod([0]), Mock())
        bbr = BatchBackwardRunner(Mock(), Mock(), test_prover_params, "decoder_type")

        to_check = Mock()
        bbr.expansion_handler = MockExpansionHandler(
            ready=[
                [],  # first, return nothing
                [(0, to_check),],  # mock should be sent to prover
                [(1, Mock()),],  # with env has died, should return lean instance died
                [(2, Mock()),],  # wrong status, should asserterror
            ]
        )
        bbr.cur_proofs = {
            0: ProofRecord(MockHandler(goal=goal,)),
            1: ProofRecord(
                MockHandler(
                    goal=BackwardGoal(
                        theorem=MMTheorem(conclusion=" ", hyps=[]),
                        label="u",
                        name="u__id",
                    ),
                )
            ),
            2: ProofRecord(MockHandler(goal=goal,)),
        }
        for x in [0, 1]:
            bbr.cur_proofs[x].s._status = ProofStatus.WAITING_FOR_EXPAND

        assert bbr.send_env_expansions() == []  # nothing happens
        assert bbr.send_env_expansions() == []  # send to_check to proving env
        bbr.goal_name_to_rid["u__id"] = 42  # usually done by bbr.submit()
        bbr.expansion_handler.env.has_died.add(1)
        res = bbr.send_env_expansions()
        assert len(res) == 1
        assert res[0][0] == 42
        assert str(res[0][1][0].exception) == LeanInstanceDied
        assert 1 not in bbr.cur_proofs
        with self.assertRaises(AssertionError):
            bbr.send_env_expansions()

        bbr.cur_proofs[0].proof_handler.send_env_expansions.assert_called_once()
        bbr.cur_proofs[0].proof_handler.send_env_expansions.assert_called_with(to_check)
        bbr.cur_proofs[2].proof_handler.send_env_expansions.assert_not_called()

        bbr.close()

    def test_one_e2e(
        self,
        mocked_json_dumps,
        mocked_tail_logger,
        mocked_logger,
        mocked_gpumonitor,
        mocked_makedirs,
        mocked_handler,
        mocked_backward_init,
    ):
        """
        Check that: 
        - with all pass throughs, we recover the result
        - with an exception at any point, we raise
        """
        goal = BackwardGoal(theorem=UnMaterializedTheorem(label="to_mat"))
        the_proof = Mock()
        mocked_handler.return_value = lambda x: MockHandler(
            goal=x,
            to_expand=[Mock() for x in range(10)],
            res=ProofResult(proof=the_proof, goal=x, exception=None),
        )
        bbr = BatchBackwardRunner(Mock(), Mock(), test_prover_params, "decoder_type")

        bbr.env = MockBackwardEnv()
        bbr.expansion_handler = MockExpansionHandler(
            pre_ready={0: [Mock() for x in range(10)]}
        )

        bbr.start()
        bbr.submit(goal)
        while True:
            output = bbr.ready()
            if len(output) > 0:
                break
        assert output[0][0] == 0 and output[0][1][0].proof == the_proof
        assert bbr.iterate == 10
        bbr.env.finish_goal.assert_called_once()
        bbr.close()

    def test_multiple_e2e(
        self,
        mocked_json_dumps,
        mocked_tail_logger,
        mocked_logger,
        mocked_gpumonitor,
        mocked_makedirs,
        mocked_handler,
        mocked_backward_init,
    ):
        """
        Check that: 
        - with all pass throughs, we recover the result
        - with an exception at any point, we raise
        """
        n_total = 17
        simu = 5
        proofs = []
        goals = []
        to_prove = {}
        expected_iterates = []
        for i in range(n_total):
            proofs.append(Mock())
            g = BackwardGoal(theorem=UnMaterializedTheorem(label=f"{i}"))
            to_prove[g.name] = (g, i + 1, proofs[-1])
            goals.append(g)
            if len(expected_iterates) < simu:
                offset = 0
            else:
                offset = expected_iterates[-simu]
            expected_iterates.append(i + 1 + offset)

        def get_mock_handler(g: BackwardGoal):
            g, n, proof = to_prove[g.name]
            return MockHandler(
                goal=g,
                to_expand=[Mock() for x in range(n)],
                res=ProofResult(proof=proof, goal=g, exception=None),
            )

        mocked_handler.return_value = get_mock_handler
        bbr = BatchBackwardRunner(Mock(), Mock(), test_prover_params, "decoder_type")

        bbr.env = MockBackwardEnv()
        bbr.expansion_handler = MockExpansionHandler(
            pre_ready={i: [Mock() for _ in range(i + 1)] for i in range(n_total)}
        )
        with closing(bbr):
            it = make_iterator(bbr, max_in_worker=simu, input_it=iter(goals))
            for i, output in enumerate(it):
                proof_result = output[0]
                iterate = bbr.iterate
                assert proof_result.proof == proofs[i], proof_result.goal.name
                assert iterate == expected_iterates[i]
            assert bbr.iterate == expected_iterates[-1]
            assert len(bbr.env.finish_goal.mock_calls) == n_total

    def test_iterate_e2e(
        self,
        mocked_json_dumps,
        mocked_tail_logger,
        mocked_logger,
        mocked_gpumonitor,
        mocked_makedirs,
        mocked_handler,
        mocked_backward_init,
    ):
        """
        Check that: 
        - with all pass throughs, we recover the result
        - with an exception at any point, we raise
        """
        n_total = 17
        simu = 5
        proofs = []
        goals = []
        to_prove = {}
        expected_iterates = []
        for i in range(n_total):
            proofs.append(Mock())
            g = BackwardGoal(theorem=UnMaterializedTheorem(label=f"{i}"))
            to_prove[g.name] = (g, i + 1, proofs[-1])
            goals.append(g)
            if len(expected_iterates) < simu:
                offset = 0
            else:
                offset = expected_iterates[-simu]
            expected_iterates.append(i + 1 + offset)

        def get_mock_handler(g: BackwardGoal):
            g, n, proof = to_prove[g.name]
            return MockHandler(
                goal=g,
                to_expand=[Mock() for x in range(n)],
                res=ProofResult(proof=proof, goal=g, exception=None),
            )

        mocked_handler.return_value = get_mock_handler
        bbr = BatchBackwardRunner(Mock(), Mock(), test_prover_params, "decoder_type")

        bbr.env = MockBackwardEnv()
        bbr.expansion_handler = MockExpansionHandler(
            pre_ready={i: [Mock() for _ in range(i + 1)] for i in range(n_total)}
        )
        with closing(bbr):
            it = make_iterator(bbr, max_in_worker=simu, input_it=iter(goals))
            for i, output in enumerate(it):
                proof_result = output[0]
                iterate = bbr.iterate
                assert proof_result.proof == proofs[i], proof_result.goal.name
                assert iterate == expected_iterates[i]
            assert bbr.iterate == expected_iterates[-1]
            assert len(bbr.env.finish_goal.mock_calls) == n_total
