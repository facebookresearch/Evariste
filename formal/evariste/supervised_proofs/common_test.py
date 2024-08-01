# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import importlib.resources as pkg_resources

from evariste import json
from evariste.supervised_proofs import resources
from evariste.supervised_proofs.common import ProofWithStatement


def test_proof_serialization():
    data = pkg_resources.read_text(resources, "test_proof.json")
    proof_dict = json.loads(data)

    proof = ProofWithStatement.from_dict(proof_dict)

    assert proof.to_dict() == proof_dict
