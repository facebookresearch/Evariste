# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import typer
from typing import Optional, Set, Callable, Dict, Type
from collections import defaultdict

from evariste.logger import create_logger
from params.params import ConfStore
from evariste.model.utils import load_from_checkpoint
from evariste.model.data.dictionary import Dictionary
from evariste.model.data.envs.metamath import MetamathDataEnvironment
from evariste.model.data.envs.hol_light import HOLLightDataEnvironment
from evariste.model.transformer import TransformerModel
from evariste.model.transformer_args import DecodingParams

from evariste.syntax.parser import get_parser, Parser

from evariste.backward.graph import Theorem, MalformedTheorem
from evariste.backward.env.metamath.graph import MMTheorem
from evariste.backward.env.hol_light.graph import HLTheorem


DEVICE = "cuda"


logger = create_logger(None, 0)


def mm_valid_statement_syntax(s: str, parser: Parser) -> bool:
    assert type(s) is str
    toks = s.split()
    if toks[0] != "|-":
        return False
    return parser.has_valid_syntax(["wff", *toks[1:]])


def mm_valid_theorem_syntax(theorem: MMTheorem, parser: Parser) -> bool:
    for _, hyp in theorem.hyps:
        if not mm_valid_statement_syntax(hyp, parser):
            return False
    return mm_valid_statement_syntax(theorem.conclusion, parser)


def load_mm_theorems(dataset: str = "new3") -> Set[MMTheorem]:
    """
    Load Metamath theorems.
    """
    logger.info(f"Loading Metamath theorems ...")
    logger.disabled = True

    # build env
    dico_env = Dictionary.create_empty()
    params_env = ConfStore["default_cfg"]
    params_env.mm.dataset = ConfStore[dataset]
    params_env.tasks = "mm_x2y_informallatex--formalmm_seq2seq"
    env = MetamathDataEnvironment(dico_env, params_env)

    # load theorems
    mm_theorems = set()
    for split in ["train", "valid", "test"]:
        for seq in env.get_sequences(split):
            mm_theorems.add(MMTheorem.from_tokens(seq))

    logger.disabled = False
    logger.info(f"Found {len(mm_theorems)} unique Metamath theorems.")
    return mm_theorems


def load_hl_theorems(dataset: str = "hl_complex_multivariate_new") -> Set[HLTheorem]:
    """
    Load HOL-Light theorems.
    """
    logger.info(f"Loading HOL-Light theorems ...")
    logger.disabled = True

    # build env
    dico_env = Dictionary.create_empty()
    params_env = ConfStore["default_cfg"]
    params_env.hl.dataset = ConfStore[dataset]
    params_env.tasks = "hl_x2y_goal--tactic_seq2seq"
    env = HOLLightDataEnvironment(dico_env, params_env)

    # load theorems
    hl_theorems = set()
    for split in ["train", "valid", "test"]:
        for seq in env.get_sequences(split):
            hl_theorems.add(HLTheorem.from_tokens(seq))

    logger.disabled = False
    logger.info(f"Found {len(hl_theorems)} unique HOL-Light theorems.")
    return hl_theorems


def load_model(path):
    """
    Reload model.
    """
    assert os.path.isfile(path)
    modules, dico, params = load_from_checkpoint(
        path=path, module_names=["encoder"], device=DEVICE, fp16=False
    )
    encoder = modules["encoder"]
    encoder.cuda()
    encoder.eval()
    id2lang = dict(enumerate(sorted(["hl", "mm", "arxivstatements"])))
    lang2id = {lang: i for i, lang in id2lang.items()}
    return encoder, dico, params, lang2id


def _sample_theorems(
    encoder: TransformerModel,
    dico: Dictionary,
    lang_id: int,
    decoding_params: DecodingParams,
    n_generations: int,
    max_attempts: int,
    output_path: str,
    th_builder: Type[Theorem],
    existing_theorems: Set[Theorem],
    has_valid_syntax: Callable,
):
    # stats
    n_total = 0
    n_malformed = 0
    n_invalid_syntax = 0
    n_valid = 0
    old_theorems: Dict[Theorem, int] = defaultdict(
        int
    )  # generated theorems from the database
    new_theorems: Dict[Theorem, int] = defaultdict(
        int
    )  # generated theorems not in the database

    f = open(output_path, "w")

    while n_total < max_attempts:

        # generate
        gen_tokens, gen_len, gen_scores = encoder.generate(
            src_enc=None,
            src_len=None,
            decoding_params=decoding_params,
            langs=lang_id,
            forbidden_tokens=None,
        )

        bs = decoding_params.n_samples
        assert len(gen_len) == bs

        for i in range(bs):

            # enough attemps
            if n_total >= max_attempts:
                break

            toks = [dico[j] for j in gen_tokens[i, : gen_len[i]].tolist()]
            assert toks[0] == toks[-1] == "</s>"
            toks = toks[1:-1]

            n_total += 1

            # parse theorem
            try:
                theorem = th_builder.from_tokens(toks)
            except MalformedTheorem as e:
                n_malformed += 1
                continue

            # check syntax
            if not has_valid_syntax(theorem):
                n_invalid_syntax += 1
                continue

            # export theorem
            if theorem not in existing_theorems and theorem not in new_theorems:
                f.write(" ".join(theorem.tokenize()) + "\n")
                f.flush()

            # update stats
            n_valid += 1
            if theorem in existing_theorems:
                old_theorems[theorem] += 1
            else:
                new_theorems[theorem] += 1

            # enough valid theorems
            if len(new_theorems) >= n_generations:
                break

        # stats
        logger.info(
            f"Generated {n_valid}/{n_total} valid theorems. {n_malformed} malformed, "
            f"{n_invalid_syntax} with invalid syntax, "
            f"{sum(old_theorems.values())} old theorems ({len(old_theorems)} unique), "
            f"{sum(new_theorems.values())} new theorems ({len(new_theorems)} unique)."
        )

        # enough valid theorems
        if len(new_theorems) >= n_generations:
            break

    f.close()
    logger.info(f"Exported {len(new_theorems)} new theorems to {output_path}")


def sample_theorems(
    data_type: str = "mm",
    model_path: str = "",
    batch_size: int = 16,
    fixed_temperature: float = 1,
    top_k: Optional[int] = None,
    max_gen_len: int = 128,
    n_generations: int = 1000,
    max_attempts: int = 3000,
    output_path: str = "",
):
    assert data_type in ["mm", "hl"]
    assert not os.path.isfile(output_path)
    assert n_generations < max_attempts

    # decoding parameters
    decoding_params = DecodingParams(
        use_beam=False,
        early_stopping=True,
        max_gen_len=max_gen_len,
        n_samples=batch_size,
        fixed_temperature=fixed_temperature,
        top_k=top_k,
        stop_symbol="</s>",
        use_cache=True,
    )
    logger.info(
        f"Data: {data_type}\nModel: {model_path}\nDecoding parameters: {decoding_params}\n"
        f"N generations: {n_generations}\nOutput path: {output_path}"
    )

    # load model
    logger.info(f"Reloading model from {model_path} ...")
    encoder, dico, _, lang2id = load_model(model_path)

    # Metamath
    if data_type == "mm":
        th_builder: Type[Theorem] = MMTheorem
        existing_theorems: Set[Theorem] = load_mm_theorems()  # type: ignore  # Set is invariant...
        parser = get_parser("new3")

        def has_valid_syntax(theorem):
            return mm_valid_theorem_syntax(theorem, parser)

    # HOL-Light
    elif data_type == "hl":
        th_builder = HLTheorem
        existing_theorems = load_hl_theorems()  # type: ignore

        def has_valid_syntax(theorem):
            # TODO: implement
            return True

    _sample_theorems(
        encoder=encoder,
        dico=dico,
        lang_id=lang2id[data_type],
        decoding_params=decoding_params,
        n_generations=n_generations,
        max_attempts=max_attempts,
        output_path=output_path,
        th_builder=th_builder,
        existing_theorems=existing_theorems,
        has_valid_syntax=has_valid_syntax,
    )


if __name__ == "__main__":
    typer.run(sample_theorems)
