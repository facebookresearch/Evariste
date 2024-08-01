# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from pathlib import Path
from typing import List

from evariste.comms.store import AnnotatedGeneration
from evariste.comms.zip_store import ChunkedZipStore
from evariste.forward.online_generation.worker_type import WorkerType


def load_last_logged_generations(
    exp_path_str: str, is_prover: bool = True, n_chunks: int = 1
) -> List[AnnotatedGeneration]:
    exp_path = Path(exp_path_str)
    actor_path = (
        exp_path / f"{WorkerType.PROVER_ACTOR}"
        if is_prover
        else exp_path / f"{WorkerType.GENERATOR_ACTOR}"
    )
    assert actor_path.exists(), actor_path
    gen_paths = actor_path / "generations_0"
    assert gen_paths.exists(), f"{gen_paths} not in {list(actor_path.iterdir())}"
    chunks = ChunkedZipStore(root_path=gen_paths)
    chunk_ids = chunks.ready_chunks()
    chunk_ids = chunk_ids[-n_chunks:]
    print(f"Loading chunks: {chunk_ids} from {gen_paths}")
    gens = []
    for chunk_id in chunk_ids:
        store = chunks.get_chunk(chunk_id)
        gens.extend(store.read_pickle_zip("sequences"))
    return gens
