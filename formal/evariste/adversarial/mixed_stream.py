# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Optional, Union, Tuple, List, Iterator
import random

from params import ConfStore
from evariste.datasets.metamath import MetamathDatasetConf
from evariste.trainer.args import TrainerArgs
from evariste.model.data.dictionary import Dictionary
from evariste.model.data.envs.metamath import MetamathDataEnvironment
from evariste.forward.common import GenerationError
from evariste.forward.proof_search import (
    GenerationHistory,
    StandardProofSearch,
    ForwardProofSearch,
)


ForwardProofStream = Iterator[Tuple[int, Union[ForwardProofSearch, GenerationError]]]


def mixed_stream(
    generation_mix: List[Tuple[float, str]],
    generator: Optional[ForwardProofStream] = None,
) -> Iterator[Tuple[int, str, GenerationHistory]]:
    if "generator" in {x[1] for x in generation_mix} and generator is None:
        raise RuntimeError(f"generator cannot be None for mix: {generation_mix}")
    generators = []
    weights: List[float] = []
    for weight, name in generation_mix:
        weights.append(weight)
        if name == "generator":
            assert generator is not None
            generators.append((name, generator))
        elif name.startswith("dataset_"):
            _d, lang, *cfg_str_ = name.split("_")
            cfg_str = "_".join(cfg_str_)
            if lang == "mm":
                assert isinstance(ConfStore[cfg_str], MetamathDatasetConf)
                tr: TrainerArgs = ConfStore["default_cfg"]
                tr.mm.dataset = ConfStore[cfg_str]
                tr.tasks = "mm_fwd_seq2seq"
                dataset = MetamathDataEnvironment(Dictionary.create_empty(), tr)
                # TODO: fix type
                generators.append((name, dataset.get_graph_iterator(split="train")))  # type: ignore
            else:
                raise RuntimeError(f"Language {lang} not handled yet")
        else:
            raise RuntimeError(f"Misunderstood stream name {name}")
    i = 0
    while True:
        # sample a generator according to the specified weights
        src, generator = random.choices(generators, weights, k=1)[0]
        generated = next(generator)[1]
        if src == "generator":
            assert isinstance(generated, StandardProofSearch)
            # TODO: fix type
            generated: GenerationHistory = generated.generation  # type: ignore
        assert isinstance(generated, GenerationHistory)
        yield i, src, generated
        i += 1
