# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from collections import defaultdict, OrderedDict
import gc
from typing import Optional, Tuple, Iterator, List, Dict
from logging import getLogger
from pathlib import Path
from evariste.backward.prover.utils import GPUMonitor

from torch.distributed import ReduceOp
from torch.distributed import all_reduce
from zmq import ZMQError
import io
import os
import zmq
import time
import shutil
import itertools
import subprocess
import numpy as np
import sklearn.metrics

import torch
import torch.distributed
from torch.nn import functional as F

import submitit
from submitit.core.utils import FailedJobError, FailedSubmissionError

from evariste.adversarial_offline.generator import OfflineGenerator, GeneratorArgs
from evariste.backward.model.beam_search_kind import BeamSearchKind
from params import ConfStore
from evariste import json as json
from evariste.utils import COND_TOK, logged_closing
from evariste.datasets import DatasetConf
from evariste.comms.zmq import ZMQNotReadySample
from evariste.clusters.utils import clusterify_partitions
from evariste.envs.hl.tokenizer import detokenize_hl
from evariste.trainer.args import TrainerArgs
from evariste.backward.graph import get_proof_size, get_proof_depth
from evariste.backward.goal_factory import get_goals_to_prove
from evariste.backward.prover.args import MCTSParams
from evariste.backward.prover.bwd_prove import bwd_prove
from evariste.backward.prover.prover import (
    ConditioningKind,
    ProverParams,
    ProverKind,
)
from evariste.backward.prover.zmq_prover import launch_async, ZMQProverParams
from evariste.model.transformer_utils import get_clm_mask_target
from evariste.model.utils import to_cuda, create_subseq_masks, get_path_bwd_proving_eval
from evariste.model.trainer import Trainer
from evariste.model.transformer_args import DecodingParams
from evariste.datasets.lean import LeanDatasetConf

from leanml import DeadLean


BLEU_SCRIPT_PATH = os.path.join(
    os.path.abspath(os.path.dirname(__file__)), "multi-bleu.perl"
)
TRAIN_PRED_EVAL_SIZE = 300
BT_MAX_EVAL_SIZE = 500
N_EVAL_GENERATIONS = 500  # using TrainParams.n_th_to_prove instead


logger = getLogger()


FWD_PROVER_KEY = "fwd_prover"


def evaluate_binary_predictions(results, scores, split, task):
    """
    Evaluate a list of metrics for model binary predictions.
    `results` is a list of (th_name, [pos_prob, neg_prob], target).
    """
    n_valid = defaultdict(int)
    n_total = defaultdict(int)

    for name, (p1, p0), y in results:
        assert y in [0, 1]
        assert 0 <= p0 + p1 <= 1 + 1e-5, p0 + p1
        n_valid[name] += (p1 >= 0.5) == y
        n_total[name] += 1

    # sequence prediction accuracy
    cor = sum(n_valid.values())
    tot = sum(n_total.values())
    acc = 100.0 * cor / tot
    logger.info(f"{task} ({split}) - Sequence accuracy: {acc:.3}% ({cor}/{tot})")
    scores[f"{split}-{task}-tok-acc"] = acc

    # theorem prediction accuracy
    th_names = set(n_total.keys())
    cor = sum(n_valid[x] == n_total[x] for x in th_names)
    tot = len(th_names)
    acc = 100.0 * cor / tot
    logger.info(f"{task} ({split}) - Theorem accuracy: {acc:.3}% ({cor}/{tot})")
    scores[f"{split}-{task}-trm-acc"] = acc

    # ROC AUC score
    roc_auc_score = sklearn.metrics.roc_auc_score(
        y_true=[y for _, _, y in results], y_score=[p1 for _, (p1, _), _ in results]
    )
    logger.info(f"{task} ({split}) - ROC AUC score: {roc_auc_score}")
    scores[f"{split}-{task}-roc-auc-score"] = roc_auc_score


def compute_usage(counts: np.ndarray) -> Tuple[float, float]:

    # usage
    usage = (counts != 0).sum() / len(counts)

    # compute entropy
    p = counts / counts.sum()
    p[p == 0] = 1
    entropy = -(np.log(p) * p).sum()

    return float(usage), float(entropy)


def evaluate_diversity(sentences: List[List[int]], dico):
    """
    Evaluate the diversity of generated sentences.
    """
    counts = np.zeros((len(dico),), dtype=np.float64)
    for sent in sentences:
        for wid in sent:
            counts[wid] += 1

    usage, entropy = compute_usage(counts)

    # average length
    avg_len = np.mean([len(sent) for sent in sentences])

    return usage, entropy, avg_len


class Evaluator(object):
    def __init__(self, trainer: Trainer):
        """
        Initialize evaluator.
        """
        self.trainer = trainer
        self.modules = trainer.modules
        self.envs = trainer.envs
        self.params: TrainerArgs = trainer.params
        self.dico = trainer.dico
        self.metrics_logger = trainer.metrics_logger
        self._cache: Dict = {}

        # holds running submitit eval jobs
        # useful for "cancel_job_at_deletion" but also to check results of finished evaluations.
        self._running_eval_jobs: Dict[Tuple[str, str, int, str], submitit.Job] = dict()

        # create a directory to store hypotheses, and reference files for BLEU evaluation
        if self.params.slurm_conf.is_master:
            self.hyp_dir = os.path.join(self.params.dump_path, "hypotheses")
            os.makedirs(self.hyp_dir, exist_ok=True)

    def run_all_evals(self):
        """
        Run all evaluations.
        """
        params = self.params
        scores = OrderedDict({"epoch": self.trainer.epoch})

        with torch.no_grad():

            for split in ["train", "valid", "test"]:

                # optionally skip train / test evals
                if self.params.no_eval_on_train and split == "train":
                    logger.warning("Skipping eval on train (see no_eval_on_train)")
                    continue
                if self.params.no_eval_on_test and split == "test":
                    logger.warning("Skipping eval on test (see no_eval_on_test)")
                    continue

                # predict commands / goals
                for task in params.parsed_tasks():
                    if task.endswith("_clm"):
                        self.evaluate_clm_mlm(scores, split, task, causal=True)
                    elif task.endswith("_mlm"):
                        self.evaluate_clm_mlm(scores, split, task, causal=False)
                    elif task.endswith("_mass"):
                        self.evaluate_mass(scores, split, task)
                    elif task.endswith("_cclm"):
                        self.evaluate_cclm(scores, split, task)
                        self.evaluate_clm_mlm(
                            scores, split, task, causal=True, cclm=True
                        )
                    elif task.endswith("_seq2seq"):
                        eval_bleu = params.eval_bleu and params.slurm_conf.is_master
                        self.evaluate_seq2seq(scores, split, task, eval_bleu)
                        if params.beam_eval:
                            self.evaluate_seq2seq_beam(scores, split, task)
                    elif task.endswith("distillation"):
                        # TODO evaluate mcts
                        if "mcts" in task:
                            continue
                        eval_bleu = params.eval_bleu and params.slurm_conf.is_master
                        if params.distillation.online or self.trainer.epoch < 1:
                            # evaluate model with big decoder (only at epoch zero for offline distillation as big model is static)
                            self.evaluate_seq2seq(
                                scores, split, task + "-bigdec", eval_bleu
                            )
                        # evaluate model
                        self.evaluate_seq2seq(scores, split, task, eval_bleu)
                    elif task.endswith("_seq2tok"):
                        self.evaluate_seq2tok_seq2seqtok(
                            scores, split, task, with_decoder=False
                        )
                    elif task.endswith("_seq2seqtok"):
                        self.evaluate_seq2tok_seq2seqtok(
                            scores, split, task, with_decoder=True
                        )
                    elif task.endswith("_seq2emb"):
                        self.evaluate_seq2emb(scores, split, task)
                    elif task.endswith("_bt"):
                        self.evaluate_bt(scores, split, task)
                    elif task.endswith("_disc"):
                        self.evaluate_seq2seq_discriminator(scores, split, task)
                        if not self.params.gan.fixed_generator:
                            self.evaluate_seq2seq(
                                scores,
                                split,
                                task,
                                eval_bleu=False,
                                use_dicriminator=True,
                            )
                    elif "mcts_critic" in task:
                        logger.warning(f"Skipping {split} eval for MCTS")
                        # TODO restore test only and fix bug for valid
                        # if split == "test" or split == "valid":
                        # else:
                        #     self.evaluate_mcts_critic(scores, split, task)
                    elif "mcts_tactic" in task or "mcts_minproof" in task:
                        logger.warning(f"Skipping {split} eval for MCTS")
                        # TODO restore test only and fix bug for valid
                        # if split == "test" or split == "valid":
                        # else:
                        #     self.evaluate_mcts_s2s(scores, split, task)
                    elif "rl" in task or "mcts_effect" in task:
                        pass
                    else:
                        raise Exception(f"Unknown task: {task}")

            # proving evaluations
            for env_name, split in params.fwd_proving_eval:
                self.evaluate_fwd_prover(scores, env_name, split)
            for env_name, split in params.bwd_proving_eval:
                if any("_distillation" in task for task in params.parsed_tasks()) and (
                    params.distillation.online or self.trainer.epoch < 1
                ):
                    self.evaluate_bwd_prover(
                        scores, env_name, split, decoder_type="big_decoder"
                    )
                self.evaluate_bwd_prover(
                    scores, env_name, split, decoder_type="decoder"
                )

        if self.trainer.stat_socket is not None:
            try:
                self.trainer.stat_socket.send_json(
                    {"type": "trainer_eval", "scores": scores}, zmq.NOBLOCK
                )
            except ZMQError as e:
                logger.warning(f"Got {e} while sending trainer_eval over stat_socket!")

        self.metrics_logger.log_metrics(
            {k.replace("-", "/"): v for k, v in scores.items()}
        )

        return scores

    def evaluate_clm_mlm(self, scores, split, task, causal, cclm=False):
        """
        CLM / MLM evaluation.
        """
        assert split in ["train", "valid", "test"]
        assert type(causal) is bool
        assert type(cclm) is bool
        assert not cclm or causal and task.endswith("_cclm")
        params = self.params
        encoder = self.modules["encoder"]
        encoder.eval()
        encoder = encoder.module if params.slurm_conf.multi_gpu else encoder

        logger.info(f"===== Evaluating {task} ({split} set) =====")

        # create iterator
        env_name = task.split("_")[0]
        iterator = self.envs[env_name].create_data_loader(split, task)
        if split == "train":
            iterator = itertools.islice(
                iterator, TRAIN_PRED_EVAL_SIZE // params.batch.size
            )

        # deterministic random generator to always mask the same words during eval
        if not causal:
            rng = np.random.RandomState(0)

        # token stats
        xe_loss = 0
        n_valid_tok = 0
        n_total_tok = 0

        for batch in iterator:

            # generate batch / select words to predict
            x = batch["x"]
            xlen = batch["xlen"]
            langs = batch.get("langs", None)

            # cclm
            if cclm:
                discr = torch.full(size=(len(xlen),), fill_value=1)
            else:
                discr = None

            # tokens to predict / cuda
            if causal:
                pred_mask, target = get_clm_mask_target(x, xlen)
            else:
                x, pred_mask, target = self.trainer.get_mlm_mask_target(
                    x, xlen, rng=rng
                )
            x, xlen, pred_mask, target = to_cuda(x, xlen, pred_mask, target)
            langs, discr = to_cuda(langs, discr)

            # forward
            tensor = encoder(
                "fwd", causal=causal, tokens=x, lengths=xlen, langs=langs, discr=discr
            )

            # compute loss / optimize
            word_scores, loss = encoder(
                "compute_loss",
                tensor=tensor,
                pred_mask=pred_mask,
                target=target,
                reduction="none",
                # epsilon=params.label_smoothing_eps,
            )

            # update token stats
            correct_pred = word_scores.max(1)[1] == target
            xe_loss += loss.sum().item()
            n_valid_tok += correct_pred.long().sum().item()
            n_total_tok += len(target)

        # results (token level)
        ppl = np.exp(xe_loss / n_total_tok)
        acc_tok = 100.0 * n_valid_tok / n_total_tok
        logger.info(
            f"{task} - {n_valid_tok}/{n_total_tok} ({acc_tok:.3}%) "
            f"correctly predicted {split} tokens. Perplexity: {ppl:.3}"
        )
        scores[f"{split}-{task}-tok-ppl"] = ppl
        scores[f"{split}-{task}-tok-acc"] = acc_tok

    def evaluate_mass(self, scores, split, task):
        """
        CLM / MLM evaluation.
        """
        assert split in ["train", "valid", "test"]
        params = self.params
        encoder = self.modules["encoder"]
        decoder = self.modules["decoder"]
        encoder.eval()
        decoder.eval()
        encoder = encoder.module if params.slurm_conf.multi_gpu else encoder
        decoder = decoder.module if params.slurm_conf.multi_gpu else decoder

        logger.info(f"===== Evaluating {task} ({split} set) =====")

        # create iterator
        env_name = task.split("_")[0]
        if split == "train":
            iterator = itertools.islice(
                self.trainer.iterators[task], TRAIN_PRED_EVAL_SIZE // params.batch.size
            )
        else:
            iterator = self.envs[env_name].create_data_loader(split, task)

        # deterministic random generator to always mask the same words during eval
        rng = np.random.RandomState(0)

        # token stats
        xe_loss = 0
        n_valid_tok = 0
        n_total_tok = 0

        for batch in iterator:

            # generate batch
            (x, xlen), (y, ylen), positions = self.trainer.get_mass_batch(
                batch["x"], batch["xlen"], rng=rng
            )

            # tokens to predict
            pred_mask, target = get_clm_mask_target(y, ylen)

            # decoder does not attend masked tokens
            enc_mask = x.ne(params.dico.mask_index)

            # cuda
            x, xlen, y, ylen, positions = to_cuda(x, xlen, y, ylen, positions)
            enc_mask, pred_mask, target = to_cuda(enc_mask, pred_mask, target)

            # encode x
            encoded = encoder("fwd", causal=False, tokens=x, lengths=xlen)
            if encoder.is_changing_input_lengths():
                encoded_len = encoder.new_input_lengths(xlen)
                # TODO: maybe adapt enc_mask to match kernel conv
                enc_mask = None  # shape mismatch between encoded and enc_mask
            else:
                encoded_len = xlen

            # decode y
            decoded = decoder(
                "fwd",
                causal=True,
                tokens=y,
                lengths=ylen,
                src_enc=encoded,
                src_len=encoded_len,
                positions=positions,
                enc_mask=enc_mask,
            )

            # compute loss / optimize
            word_scores, loss = decoder(
                "compute_loss",
                tensor=decoded,
                pred_mask=pred_mask,
                target=target,
                reduction="none",
                # epsilon=params.label_smoothing_eps,
            )

            # update token stats
            correct_pred = word_scores.max(1)[1] == target
            xe_loss += loss.sum().item()
            n_valid_tok += correct_pred.long().sum().item()
            n_total_tok += len(target)

        # results (token level)
        ppl = np.exp(xe_loss / n_total_tok)
        acc_tok = 100.0 * n_valid_tok / n_total_tok
        logger.info(
            f"{task} - {n_valid_tok}/{n_total_tok} ({acc_tok:.3}%) "
            f"correctly predicted {split} tokens. Perplexity: {ppl:.3}"
        )
        scores[f"{split}-{task}-tok-ppl"] = ppl
        scores[f"{split}-{task}-tok-acc"] = acc_tok

    def evaluate_cclm(self, scores, split, task):
        """
        CCLM evaluation.
        """
        assert split in ["train", "valid", "test"]
        params = self.params

        discriminator = self.modules["discriminator"].eval()
        classifier = self.modules["classifier"].eval()
        if params.slurm_conf.multi_gpu:
            discriminator = discriminator.module
            classifier = classifier.module

        logger.info(f"===== Evaluating {task} ({split} set) =====")

        N_EVAL_CCLM = 2000 if split == "valid" else 100

        for label, score in [("real", 1), ("fake", 0)]:

            generations = []
            predictions = []

            while len(generations) < N_EVAL_CCLM:

                # generate samples
                x, xlen = self.trainer.generate_samples(score=score)
                for i in range(len(xlen)):
                    generations.append(x[i, : xlen[i]].tolist())

                # classify samples
                tensor = discriminator("fwd", causal=False, tokens=x, lengths=xlen)
                predicted = classifier(tensor[:, 0])
                assert predicted.size() == (len(xlen), 1)
                predictions.extend(torch.sigmoid(predicted.view(-1)).tolist())

            assert len(generations) == len(predictions)

            # export generations
            if self.params.slurm_conf.is_master:
                fname = f"cclm.{split}.{label}.{scores['epoch']}"
                fpath = os.path.join(self.hyp_dir, fname)
                with open(fpath, "w") as f:
                    for sent, score in zip(generations, predictions):
                        words = [self.dico[wid] for wid in sent]
                        sent = " ".join(words)
                        f.write(f"{score:.3f} {sent}\n")
                logger.info(
                    f"Exported {len(generations)} {split} generations to {fpath}"
                )

            # evaluate generations
            usage, entropy, avg_len = evaluate_diversity(generations, self.dico)

            # update scores
            accuracy = np.mean([int(x < 0.5) for x in predictions])
            scores[f"{split}-{task}-{label}-usage"] = usage
            scores[f"{split}-{task}-{label}-entropy"] = entropy
            scores[f"{split}-{task}-{label}-avg_len"] = avg_len
            scores[f"{split}-{task}-{label}-disc_acc"] = accuracy
            scores[f"{split}-{task}-{label}-disc_mean_pred"] = np.mean(predictions)

    def evaluate_seq2seq(
        self, scores, split, task, eval_bleu, use_dicriminator: bool = False
    ):
        """
        Evaluate perplexity and next word prediction accuracy.
        Perfect match evaluation.
        """

        # TODO: replace with corresponding eval
        if "subproof_online_mcts" in task:
            return

        if "fwd_error_cond" in task:
            return

        assert split in ["train", "valid", "test"]
        params = self.params
        encoder = self.modules["encoder"]
        if task.endswith("-bigdec"):
            assert "distillation" in task and "big_decoder" in self.modules
            task_for_data = task[: -len("-bigdec")]
            decoder = self.modules["big_decoder"]
        else:
            task_for_data = task
            decoder = self.modules["decoder"]

        multi_gpu = params.slurm_conf.multi_gpu
        encoder.eval()
        decoder.eval()
        encoder = encoder.module if multi_gpu else encoder
        decoder = decoder.module if multi_gpu else decoder

        logger.info(f"===== Evaluating {task} ({split} set) =====")

        # create iterator
        env_name = task_for_data.split("_")[0]
        if split == "train":
            iterator = itertools.islice(
                self.trainer.iterators[task], TRAIN_PRED_EVAL_SIZE // params.batch.size
            )
        else:
            iterator = self.envs[env_name].create_data_iterator(
                split, task, try_fetch=True
            )

        # token stats
        n_valid_tok = 0
        n_total_tok = 0

        # sequence stats
        n_valid_seq: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        n_total_seq: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # class conditioning
        if "x2y" in task and params.cond.n_classes > 0:
            all_classes = []

        # model outputs / references
        if eval_bleu:
            output_hyps = []
            output_refs = []

        xe_losses = []
        for batch in iterator:

            # generate batch / select words to predict
            names = batch["names"]
            x = batch["x"]
            y = batch["y"]
            xlen = batch["xlen"]
            ylen = batch["ylen"]
            langs2 = batch.get("langs2", None)
            input_conditioning = batch.get("input_conditioning", None)
            bs = len(x)

            q_conditioning = (
                params.mcts_train.q_conditioning
            )  # in ["", "sum", "prefix"]
            decoder_conditioning: Optional[torch.Tensor] = None
            if q_conditioning:
                decoder_conditioning = torch.ones(len(xlen), dtype=torch.float)

            # positions of sub-sequences
            subseq_pos = batch.get("y_subseq_pos", [{} for _ in range(bs)])
            subseq_names, subseq_masks = create_subseq_masks(subseq_pos, ylen=ylen)

            # tokens to predict
            pred_mask, target = get_clm_mask_target(y, ylen)
            assert pred_mask.eq(subseq_masks["full"].to(pred_mask)).all()

            # conditioning (Q=1)
            encoder_conditioning = xlen.clone().float() if use_dicriminator else None
            if input_conditioning is not None:
                assert encoder_conditioning is None
                encoder_conditioning = input_conditioning

            # cuda
            x, xlen, y, ylen = to_cuda(x, xlen, y, ylen)
            pred_mask, target, langs2 = to_cuda(pred_mask, target, langs2)
            encoder_conditioning, decoder_conditioning = to_cuda(
                encoder_conditioning, decoder_conditioning
            )

            # encode x
            encoded = encoder(
                "fwd",
                causal=False,
                tokens=x,
                lengths=xlen,
                discr=encoder_conditioning,
                discr_mode="" if encoder_conditioning is None else "sum",
            )
            if encoder.is_changing_input_lengths():
                xlen = encoder.new_input_lengths(xlen)
            assert xlen.max().item() <= encoded.size(1)

            if "x2y" in task and params.cond.n_classes > 0:
                n_classes = params.cond.n_classes
                q_conditioning = params.cond.input_mode

                if params.cond.enc_tgt:
                    tgt_encoder = self.modules["target_encoder"]
                    tgt_encoder.eval()
                    tgt_encoder = tgt_encoder.module if multi_gpu else tgt_encoder
                    weights, best_classes, reg_loss = tgt_encoder(
                        "fwd_tgt",
                        tokens=y,
                        lengths=ylen,
                        n_classes=n_classes,
                        hard=True,
                        T=0,
                    )
                else:
                    class_embs = self.modules["cond_embeddings"]
                    class_embs = class_embs.module if multi_gpu else class_embs
                    best_classes = decoder(
                        "best_classes_fast_split",
                        tokens=y,
                        lengths=ylen,
                        src_enc=encoded,
                        src_len=xlen,
                        n_classes=n_classes,
                        class_embs=class_embs.weight,
                        max_classes=params.cond.max_classes,
                    )
                    weights = F.one_hot(best_classes, num_classes=n_classes).float()

                assert weights.size() == (bs, n_classes)
                assert best_classes.size() == (bs,)
                with torch.cuda.amp.autocast(enabled=params.model.fp16):
                    decoder_conditioning = self.modules["cond_embeddings"](weights)
                all_classes.extend(best_classes.tolist())

            # decode y
            decoded = decoder(
                "fwd",
                causal=True,
                tokens=y,
                lengths=ylen,
                src_enc=encoded,
                src_len=xlen,
                langs=langs2,
                discr=decoder_conditioning,
                discr_mode=q_conditioning,
            )

            # loss / word scores
            word_scores, loss = decoder(
                "compute_loss",
                tensor=decoded,
                pred_mask=pred_mask,
                target=target,
                reduction="none",
                # epsilon=params.label_smoothing_eps,
            )

            # update token stats
            correct_pred = (word_scores.max(1)[1] == target).cpu()
            xe_losses += loss.tolist()
            n_valid_tok += correct_pred.long().sum().item()
            n_total_tok += len(target)

            # update sequence stats by theorem
            correct = torch.zeros_like(pred_mask, device="cpu")
            correct[pred_mask] += correct_pred
            for seq_name, mask in subseq_masks.items():
                assert len(mask) == len(names) == bs
                assert mask.size() == pred_mask.size()
                assert seq_name != "full" or mask.eq(pred_mask.cpu()).all()
                sub_cor = correct & mask
                # print(
                #     seq_name,
                #     [
                #         self.dico[j]
                #         for j in y.cpu()[:, 1:].masked_select(mask[:, :-1]).tolist()
                #     ],
                # )

                # print(seq_name, mask.shape)
                # print("ws", word_scores.shape, "tgt", target.shape)
                # print("correct", correct.shape, "sc", sub_cor.shape, "y", y.shape)
                # print("pred_mask", pred_mask.shape)
                # print([self.dico[j.item()] for j in target.cpu()])
                # print("\n\n\n")
                # print([self.dico[j.item()] for j in word_scores.max(1)[1].cpu()])
                # print("-------")

                valid_seq = (sub_cor.sum(1) == mask.sum(1)).cpu().long()

                # print()

                for valid, trm_name in zip(valid_seq.tolist(), names):
                    n_valid_seq[seq_name][trm_name] += valid
                    n_total_seq[seq_name][trm_name] += 1

            # model generation
            if eval_bleu:
                decoding_params = DecodingParams(
                    max_gen_len=int(1.5 * ylen.max().item() + 10),
                    n_samples=1,
                    use_beam=False,
                )
                dec_tok, dec_len, _ = decoder.generate(
                    src_enc=encoded, src_len=xlen, decoding_params=decoding_params
                )
                output_hyps += convert_to_text(
                    dec_tok, dec_len, self.dico, params.dico.eos_index
                )
                output_refs += convert_to_text(
                    y, ylen, self.dico, params.dico.eos_index
                )

        # results (token level)
        avg_xe_loss = np.mean(xe_losses)
        std_xe_loss = np.std(xe_losses)
        ppl = np.exp(avg_xe_loss)

        acc_tok = 100.0 * n_valid_tok / n_total_tok
        logger.info(
            f"{task} - {n_valid_tok}/{n_total_tok} ({acc_tok:.3}%) "
            f"correctly predicted {split} tokens. Perplexity: {ppl:.3} // log loss {avg_xe_loss} (std: {std_xe_loss})"
        )
        scores[f"{split}-{task}-tok-ppl"] = ppl
        scores[f"{split}-{task}-tok-acc"] = acc_tok

        # results (sequence / theorem level)
        theorem_names = set(n_valid_seq["full"].keys())
        for seq_name in subseq_names:
            cor_seq = sum(n_valid_seq[seq_name].values())
            tot_seq = sum(n_total_seq[seq_name].values())
            acc_seq = 100.0 * cor_seq / tot_seq
            cor_trm = sum(
                n_valid_seq[seq_name][trm_name] == n_total_seq[seq_name][trm_name]
                for trm_name in theorem_names
            )
            tot_trm = len(theorem_names)
            acc_trm = 100.0 * cor_trm / tot_trm
            logger.info(
                f"{task} - {seq_name} {cor_seq}/{tot_seq} ({acc_seq:.3}%) "
                f"correctly predicted {split} sequences."
            )
            logger.info(
                f"{task} - {seq_name} {cor_trm}/{tot_trm} ({acc_trm:.3}%) "
                f"correctly predicted {split} theorems."
            )
            scores[f"{split}-{task}-{seq_name}-seq-acc"] = acc_seq
            scores[f"{split}-{task}-{seq_name}-trm-acc"] = acc_trm

        # class conditioning
        if "x2y" in task and params.cond.n_classes > 0:
            counts = np.zeros(params.cond.n_classes)
            for x in all_classes:
                counts[x] += 1
            usage, entropy = compute_usage(counts)
            scores[f"{split}-{task}-cond_usage"] = usage
            scores[f"{split}-{task}-cond_entropy"] = entropy

        # BLEU score
        if eval_bleu:

            # hypothesis / reference paths
            hyp_path = os.path.join(
                self.hyp_dir, f"hyp.{split}.{task}.{scores['epoch']}"
            )
            ref_path = os.path.join(self.hyp_dir, f"ref.{split}.{task}")

            # export hypotheses
            with open(hyp_path, "w", encoding="utf-8") as f:
                f.write("\n".join(output_hyps) + "\n")

            # export references
            if not os.path.isfile(ref_path):
                with open(ref_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(output_refs) + "\n")

            # evaluate BLEU score
            bleu = eval_moses_bleu(hyp_path, ref_path)
            logger.info(f"BLEU score ({split}) {hyp_path} {ref_path}: {bleu}")
            scores[f"{split}-{task}-bleu"] = bleu

    def evaluate_seq2seq_beam(self, scores, split, task):
        """
        Evaluate generations.
        Check whether the reference is in the beam.
        `hl_pred_next_tact_seq2seq` and `mm_goal2tactic_seq2seq` tasks only.
        """
        assert split in ["train", "valid", "test"]
        if task != "hl_pred_next_tact_seq2seq" and not task.startswith("mm_x2y_"):
            return

        params = self.params
        beam_params = ConfStore["decoding_bwd_eval"]

        dico = self.dico
        encoder = self.modules["encoder"]
        decoder = self.modules["decoder"]
        encoder.eval()
        decoder.eval()
        encoder = encoder.module if params.slurm_conf.multi_gpu else encoder
        decoder = decoder.module if params.slurm_conf.multi_gpu else decoder

        logger.info(f"===== Evaluating {task} ({split} set) =====")

        # create iterator
        env_name = task.split("_")[0]
        iterator = self.envs[env_name].create_data_loader(split, task)
        if split == "train":
            iterator = itertools.islice(
                iterator, TRAIN_PRED_EVAL_SIZE // params.batch.size
            )

        # generations
        outputs = []

        for bid, batch in enumerate(iterator):

            logger.info(str((bid, len(outputs))))

            # generate batch / select words to predict
            names = batch["names"]
            x = batch["x"]
            y = batch["y"]
            xlen = batch["xlen"]
            ylen = batch["ylen"]
            langs2 = batch.get("langs2", None)

            # tokens to predict / cuda
            pred_mask, target = get_clm_mask_target(y, ylen)
            x, xlen, y, ylen, pred_mask, target, langs2 = to_cuda(
                x, xlen, y, ylen, pred_mask, target, langs2
            )

            # encode x
            encoded = encoder("fwd", causal=False, tokens=x, lengths=xlen)

            # compute target loss for each sequence
            decoded = decoder(
                "fwd",
                causal=True,
                tokens=y,
                lengths=ylen,
                src_enc=encoded,
                src_len=xlen,
                langs=langs2,
            )
            _, loss = decoder(
                "compute_loss",
                tensor=decoded,
                pred_mask=pred_mask,
                target=target,
                reduction="none",
                # epsilon=params.label_smoothing_eps,
            )
            t = torch.zeros_like(pred_mask).float()
            t[pred_mask] += loss
            seq_loss = t.sum(1) / (ylen.float() - 1) ** beam_params.length_penalty

            # beam generation
            generations = decoder.generate_beam(
                encoded, xlen, decoding_params=beam_params
            )

            # store generations
            for i, (name, g_) in enumerate(zip(names, generations)):
                hyps = [(s, [dico[i] for i in hyp.tolist()[1:]]) for s, hyp in g_.hyp]
                hyps = sorted(hyps, key=lambda x: x[0], reverse=True)
                outputs.append(
                    {
                        "name": name,
                        "x": [dico[i] for i in x[i, 1 : xlen[i] - 1].tolist()],
                        "y": [dico[i] for i in y[i, 1 : ylen[i] - 1].tolist()],
                        "loss": seq_loss[i].item(),
                        "hyps": hyps,
                    }
                )

        ranks = []
        # compute MRR
        for o in outputs:
            hyps = o["hyps"]
            r = beam_params.n_samples
            for i, (_, h) in enumerate(hyps):
                if o["y"] == h:
                    r = i + 1
            ranks.append(1.0 / r)
        mrr = np.mean(ranks)
        logger.info(f"{split}-{task}-MRR = {mrr:.3} ")
        scores[f"{split}-{task}-seq-mrr"] = mrr

        # results per top-k
        for topk in sorted({1, 5, 10, 20, 50, 100, beam_params.n_samples}):

            # beam too small
            if topk > beam_params.n_samples:
                continue

            # sequence stats
            n_valid_seq = defaultdict(int)
            n_total_seq = defaultdict(int)
            for o in outputs:
                name = o["name"]
                hyps = o["hyps"]
                n_valid_seq[name] += int(any(o["y"] == h for _, h in hyps[:topk]))
                n_total_seq[name] += 1

            # results (sequence / theorem level)
            theorem_names = set(n_valid_seq.keys())
            cor_seq = sum(n_valid_seq.values())
            tot_seq = sum(n_total_seq.values())
            acc_seq = 100.0 * cor_seq / tot_seq
            cor_trm = sum(
                n_valid_seq[name] == n_total_seq[name] for name in theorem_names
            )
            tot_trm = len(theorem_names)
            acc_trm = 100.0 * cor_trm / tot_trm
            logger.info(
                f"{task} - top-k = {topk} - {cor_seq}/{tot_seq} ({acc_seq:.3}%) "
                f"correctly predicted {split} sequences."
            )
            logger.info(
                f"{task} - top-k = {topk} - {cor_trm}/{tot_trm} ({acc_trm:.3}%) "
                f"correctly predicted {split} theorems."
            )
            scores[f"{split}-{task}-seq-beam_{topk}_acc"] = acc_seq
            scores[f"{split}-{task}-trm-beam_{topk}_acc"] = acc_trm

        # output results
        output_path = os.path.join(
            params.dump_path, f"hyp.{split}.{task}.{scores['epoch']}"
        )
        with io.open(output_path, "w", encoding="utf-8") as f:
            for name in sorted(theorem_names):
                sequences = [seq for seq in outputs if seq["name"] == name]
                res = "SUCCESS" if n_valid_seq[name] == n_total_seq[name] else "FAIL"
                f.write(
                    f"========================================\n\n{name} - {len(sequences)} commands - {res}\n\n"
                )
                for seq in sequences:
                    f.write(f"==== Goalstack\n\n{detokenize_hl(seq['x'])}\n\n")
                    f.write("==== Reference\n\n")
                    f.write(f"{-seq['loss']:.5e}\t\t{detokenize_hl(seq['y'])}\n\n")
                    f.write("==== Hypotheses\n\n")
                    for s, h in seq["hyps"]:
                        res = "OK" if h == seq["y"] else ""
                        f.write(f"{s:.5e}\t{res}\t{detokenize_hl(h)}\n")
                    f.write("\n")
        logger.info(f"Generations saved to {output_path}")

    def evaluate_bt(self, scores, split: str, task: str):
        """
        Evaluate back-translation.
        For now, we only export generations.
        """
        assert split in ["train", "valid", "test"]
        params = self.params
        encoder = self.modules["encoder"]
        decoder = self.modules["decoder"]
        encoder.eval()
        decoder.eval()
        encoder = encoder.module if params.slurm_conf.multi_gpu else encoder
        decoder = decoder.module if params.slurm_conf.multi_gpu else decoder

        logger.info(f"===== Evaluating {task} ({split} set) =====")

        # retrieve languages
        lang1, lang2, lang3 = task[6:-3].split("-")
        assert lang1 == lang3 and lang1 != lang2

        # create iterator
        env_name = task.split("_")[0]
        iterator = self.envs[env_name].create_data_loader(split, task)
        iterator = itertools.islice(iterator, BT_MAX_EVAL_SIZE // params.batch.size)

        # store inputs / generations
        output_x = []
        output_y = []

        for batch in iterator:

            # generate batch / cuda
            x1 = batch["x"]
            xlen1 = batch["xlen"]
            langs2 = batch["langs2"]
            x1, xlen1, langs2 = to_cuda(x1, xlen1, langs2)

            # encode x
            encoded1 = encoder("fwd", causal=False, tokens=x1, lengths=xlen1)

            # # TODO: handle this
            # if encoder.is_changing_input_lengths():
            #     xlen = encoder.new_input_lengths(xlen)
            # assert xlen.max().item() <= encoded.size(1)

            # decoding parameters
            max_len = int(3 * xlen1.max().item() + 10)
            decoding_params = DecodingParams(
                max_gen_len=min(params.batch.max_len, max_len),
                n_samples=1,
                use_beam=False,
            )

            # translate
            x2, xlen2, _ = decoder.generate(
                src_enc=encoded1,
                src_len=xlen1,
                decoding_params=decoding_params,
                langs=langs2,
            )

            # store generations
            output_x += convert_to_text(x1, xlen1, self.dico, params.dico.eos_index)
            output_y += convert_to_text(x2, xlen2, self.dico, params.dico.eos_index)

        # export translations
        if self.params.slurm_conf.is_master:
            assert len(output_x) == len(output_y)
            out_path = os.path.join(
                self.hyp_dir, f"bt.{lang1}-{lang2}.{split}.{scores['epoch']}"
            )
            with open(out_path, "w") as f:
                for x, y in zip(output_x, output_y):
                    f.write(f"{x}\n{y}\n\n")
            logger.info(
                f"Wrote {len(output_x)} {lang1}-{lang2} translations in {out_path}"
            )

    def evaluate_seq2tok_seq2seqtok(self, scores, split, task, with_decoder):
        """
        Sequence-to-token binary classification task.
        """
        assert split in ["train", "valid", "test"]
        params = self.params
        encoder = self.modules["encoder"]
        encoder.eval()
        encoder = encoder.module if params.slurm_conf.multi_gpu else encoder
        if with_decoder:
            decoder = self.modules["decoder"]
            decoder.eval()
            decoder = decoder.module if params.slurm_conf.multi_gpu else decoder
        else:
            classifier = self.modules["classifier"]
            classifier.eval()
            classifier = (
                classifier.module if params.slurm_conf.multi_gpu else classifier
            )

        logger.info(f"===== Evaluating {task} ({split} set) =====")

        # create iterator
        env_name = task.split("_")[0]
        iterator = self.envs[env_name].create_data_iterator(split, task, try_fetch=True)
        if split == "train":
            iterator = itertools.islice(
                iterator, TRAIN_PRED_EVAL_SIZE // params.batch.size
            )

        # binary cross-entropy loss
        xe_loss = 0
        n_total = 0

        # predictions with targets
        all_predictions = []

        for batch in iterator:

            # generate batch
            names = batch["names"]
            x = batch["x"]
            y = batch["y"]
            xlen = batch["xlen"]
            bs, slen = x.size()

            # cuda / sanity checks
            x, xlen, y = to_cuda(x, xlen, y)
            assert len(names) == bs
            assert y.size() == (bs, 1)
            assert y.eq(0).sum().item() + y.eq(1).sum().item() == bs

            # encode x
            encoded = encoder("fwd", causal=False, tokens=x, lengths=xlen)
            assert encoded.size() == (bs, slen, params.model.enc_emb_dim)

            # classify
            if with_decoder:
                predicted, _ = decoder(
                    "compute_critic", src_enc=encoded, src_len=xlen, q=None
                )
            else:
                predicted = classifier(encoded[:, 0])
                predicted = predicted.log_softmax(-1)
            # sanity checks
            assert predicted.size() == (bs, 2)
            assert predicted.exp().sum(1).sub(1).abs().lt(1e-5).all()
            # compute loss: y.log x + (1-y) * log (1-x)
            target = torch.cat([y, 1 - y], 1).float()
            loss = -torch.sum(target * predicted, 1).sum()

            # store results
            xe_loss += loss.item()
            n_total += bs
            all_predictions.append(
                list(zip(names, predicted.exp().tolist(), y.view(-1).tolist()))
            )

        # merge results
        all_predictions = list(itertools.chain.from_iterable(all_predictions))
        assert len(all_predictions) == n_total

        # perplexity
        ppl = np.exp(xe_loss / n_total)
        logger.info(f"{task} ({split}) - Perplexity: {ppl:.3}")
        scores[f"{split}-{task}-tok-ppl"] = ppl

        # evaluation metrics
        evaluate_binary_predictions(all_predictions, scores, split, task)

    @torch.no_grad()
    def update_neg_sampler_embeddings(self, embedder, neg_sampler, task, scores):
        """
        Update negative sampler embeddings.
        """
        # embedding updates information
        updated = neg_sampler.last_updated[neg_sampler.last_updated >= 0].cpu().numpy()
        neg_ratio = 100.0 * len(updated) / len(neg_sampler)
        neg_mean_age = np.mean(updated)
        neg_std_age = np.std(updated)
        logger.info(
            f"{len(updated)}/{len(neg_sampler)} ({neg_ratio:.2f}%) updated embeddings. "
            f"Updated mean age: {neg_mean_age:.2f} (±{neg_std_age:.2f})"
        )
        scores[f"{task}-neg_updates-ratio"] = neg_ratio
        scores[f"{task}-neg_updates-age_mean"] = np.mean(updated)
        scores[f"{task}-neg_updates-age_std"] = np.std(updated)

        # update all embeddings
        neg_sampler.update_all_embeddings(embedder)
        # neg_sampler.update_all_embeddings_distributed(embedder)

    def evaluate_seq2emb(self, scores, split, task):
        """
        Sequence-to-embedding classification task.
        """
        assert split in ["train", "valid", "test"]
        params = self.params
        embedder = self.modules["embedder"]
        encoder = self.modules["encoder"]
        embedder.eval()
        encoder.eval()
        embedder = embedder.module if params.slurm_conf.multi_gpu else embedder
        encoder = encoder.module if params.slurm_conf.multi_gpu else encoder

        logger.info(f"===== Evaluating {task} ({split} set) =====")

        # embed all theorems
        neg_sampler = self.trainer.negative_samplers["mm"]
        if split == "train":
            self.update_neg_sampler_embeddings(embedder, neg_sampler, task, scores)
        embeddings = neg_sampler.embeddings.float()
        n_trm = len(embeddings)

        # create iterator
        env_name = task.split("_")[0]
        iterator = self.envs[env_name].create_data_loader(split, task)
        if split == "train":
            iterator = itertools.islice(
                iterator, TRAIN_PRED_EVAL_SIZE // params.batch.size
            )

        PRED_TOK_AT_K = [1, 5, 10, 25, 50, 100]

        # token stats
        xe_loss = 0
        n_valid = {k: defaultdict(int) for k in PRED_TOK_AT_K}
        n_total = defaultdict(int)

        for batch in iterator:

            # generate batch / select words to predict
            names = batch["names"]
            x = batch["x"]
            y = batch["y"]
            xlen = batch["xlen"]
            ylen = batch["ylen"]

            # retrieve target sequence IDs
            tgt_seqs = [
                [self.dico[wid] for wid in y[i, : ylen[i]].tolist()]
                for i in range(len(ylen))
            ]
            tgt_ids = [neg_sampler.seq2id[" ".join(seq)] for seq in tgt_seqs]
            tgt_ids = torch.LongTensor(tgt_ids)

            bs, slen = x.size()
            assert len(y) == len(ylen) == len(tgt_ids) == bs

            # cuda
            x, xlen, tgt_ids = to_cuda(x, xlen, tgt_ids)

            # encode x
            encoded = encoder("fwd", causal=False, tokens=x, lengths=xlen)
            assert encoded.size() == (bs, slen, params.model.enc_emb_dim)

            # compute theorem scores
            trm_scores = embeddings.mm(encoded[:, 0].transpose(0, 1)).transpose(0, 1)
            assert trm_scores.shape == (bs, n_trm)

            # compute loss
            loss = F.cross_entropy(trm_scores, tgt_ids, reduction="none")
            xe_loss += loss.sum().item()

            # update token stats
            top_matches = trm_scores.topk(max(PRED_TOK_AT_K), 1, True, sorted=True)[1]
            for k in PRED_TOK_AT_K:
                top_k_matches = (top_matches[:, :k] == tgt_ids[:, None]).sum(1)
                assert len(top_k_matches) == len(names)
                for valid, name in zip(top_k_matches.tolist(), names):
                    n_valid[k][name] += valid

            # update total token stats
            assert len(tgt_ids) == len(names)
            for name in names:
                n_total[name] += 1

        # perplexity
        ppl = np.exp(xe_loss / sum(n_total.values()))
        logger.info(f"{task} ({split}) - Perplexity: {ppl:.3}")
        scores[f"{split}-{task}-tok-ppl"] = ppl

        # token prediction accuracy @k
        for k in PRED_TOK_AT_K:
            cor = sum(n_valid[k].values())
            tot = sum(n_total.values())
            acc = 100.0 * cor / tot
            logger.info(
                f"{task} ({split}) - Token accuracy at k={k}: {acc:.3}% ({cor}/{tot})"
            )
            scores[f"{split}-{task}-tok-acc_at_{k}"] = acc

        # theorem prediction accuracy @k
        for k in PRED_TOK_AT_K:
            th_names = set(n_total.keys())
            cor = sum(n_valid[k][x] == n_total[x] for x in th_names)
            tot = len(th_names)
            acc = 100.0 * cor / tot
            logger.info(
                f"{task} ({split}) - Theorem accuracy at k={k}: {acc:.3}% "
                f"({cor}/{tot})"
            )
            scores[f"{split}-{task}-trm-acc_at_{k}"] = acc

    def evaluate_mcts_critic(self, scores, split, task):
        params = self.params
        if params.mcts_train.jsonl_data_dir == "":
            logger.info(f"===== Skipping eval for {task} ({split} set) =====")
            return
        q_conditioning = params.mcts_train.q_conditioning
        encoder = self.modules["encoder"]
        decoder = self.modules["decoder"]
        encoder.eval()
        decoder.eval()
        encoder = encoder.module if params.slurm_conf.multi_gpu else encoder
        decoder = decoder.module if params.slurm_conf.multi_gpu else decoder

        logger.info(f"===== Evaluating {task} ({split} set) =====")

        # create iterator
        env_name = task.split("_")[0]
        iterator = self.envs[env_name].create_data_loader(split, task)
        if split == "train":
            iterator = itertools.islice(
                iterator, TRAIN_PRED_EVAL_SIZE // params.batch.size
            )
        assert isinstance(next(iterator), ZMQNotReadySample)
        crits = []
        for batch in iterator:
            goal = batch["x"]
            goal_len = batch["xlen"]
            q = batch["q"]

            # cuda
            goal, goal_len, q, = to_cuda(goal, goal_len, q,)

            # encode x
            encoded = encoder("fwd", causal=False, tokens=goal, lengths=goal_len)

            critics, _ = decoder(
                "compute_critic", src_enc=encoded, src_len=goal_len, q=q
            )
            target = torch.cat([q[:, None], 1 - q[:, None]], 1)
            critic_loss = -torch.sum(
                target * critics, 1
            ).mean()  # y.log x + (1-y) * log (1-x)

            crits.append(critic_loss.item())

        # update stats
        scores[f"{split}-{task}"] = np.mean(crits)
        logger.info(f"{split}-{task}  --> {np.mean(crits)}")

    def evaluate_mcts_s2s(self, scores, split, task):
        params = self.params
        if params.mcts_train.jsonl_data_dir == "":
            logger.info(f"===== Skipping eval for {task} ({split} set) =====")
            return
        q_conditioning = params.mcts_train.q_conditioning
        encoder = self.modules["encoder"]
        decoder = self.modules["decoder"]
        encoder.eval()
        decoder.eval()
        encoder = encoder.module if params.slurm_conf.multi_gpu else encoder
        decoder = decoder.module if params.slurm_conf.multi_gpu else decoder

        logger.info(f"===== Evaluating {task} ({split} set) =====")

        # create iterator
        env_name = task.split("_")[0]
        iterator = self.envs[env_name].create_data_loader(split, task)
        if split == "train":
            iterator = itertools.islice(
                iterator, TRAIN_PRED_EVAL_SIZE // params.batch.size
            )
        assert isinstance(next(iterator), ZMQNotReadySample)
        s2s = []
        for batch in iterator:
            goal = batch["x"]
            goal_len = batch["xlen"]
            tactics = batch["y"]
            tactics_len = batch["ylen"]
            q_tactics = batch.get("q_tactics", None)
            assert (q_tactics is not None) or (q_conditioning == "")

            # cuda
            goal, goal_len, tactics, tactics_len, q_tactics = to_cuda(
                goal, goal_len, tactics, tactics_len, q_tactics
            )

            # encode x
            encoded = encoder("fwd", causal=False, tokens=goal, lengths=goal_len)
            pred_mask, target = get_clm_mask_target(tactics, tactics_len)

            # decode y
            decoded = decoder(
                "fwd",
                causal=True,
                tokens=tactics,
                lengths=tactics_len,
                src_enc=encoded,
                src_len=goal_len,
                discr=q_tactics,
                discr_mode=q_conditioning,
            )

            # compute s2s loss
            _, s2s_loss = decoder(
                "compute_loss",
                tensor=decoded,
                pred_mask=pred_mask,
                target=target,
                # epsilon=params.label_smoothing_eps,
            )

            s2s.append(s2s_loss.item())

        # update stats
        scores[f"{split}-{task}"] = np.mean(s2s)
        logger.info(f"{split}-{task}  --> {np.mean(s2s)}")

    def evaluate_fwd_prover(self, scores, env_name: str, split: str):
        from evariste.forward.common import ForwardGoal
        from evariste.forward.proof_search import StandardProofSearch
        from evariste.forward import forward_model_factory
        from evariste.forward.forward_prover import ForwardProver
        from evariste.forward.env_specifics.fwd_env_helper import FwdEnvHelper

        # only the master process takes care of evaluation
        if not self.params.slurm_conf.is_master:
            return

        # not to create dependency on forward folder for other tasks
        # [-] will crash at first evaluation

        prefix = f"fwd-{split}-{env_name}"

        metric_name = f"{prefix}-proven-greedy"

        logger.info("Beginning fwd proving")
        start = time.time()
        params = self.params
        encoder = self.modules["encoder"]
        decoder = self.modules["decoder"]
        encoder.eval()
        decoder.eval()
        encoder = encoder.module if params.slurm_conf.multi_gpu else encoder
        decoder = decoder.module if params.slurm_conf.multi_gpu else decoder

        env = self.envs[env_name]
        if env_name == "mm":
            assert split in ["valid", "minif2f_valid"]
        elif env_name == "eq":
            assert split in ["valid", "identities"]
        elif env_name == "lean":
            assert split in ["valid", "minif2f_valid"]
        elif env_name == "hol":
            assert split in ["valid", "miniF2F"]
        else:
            raise NotImplementedError

        if FWD_PROVER_KEY in self._cache:
            fwd_prover = self._cache[FWD_PROVER_KEY]
            assert isinstance(fwd_prover, ForwardProver)
        else:
            env_helper = FwdEnvHelper.from_trainer_args(
                dico=self.dico, params=self.params, env=env, env_name=env_name
            )
            fwd_prover = ForwardProver.from_trainer_args(
                params=params,
                prover_env_specifics=env_helper.get_prover_env_specifics(),
                cfg=forward_model_factory.make_prover_cfg(params, prover_type="greedy"),
                dico=self.dico,
                encoder=encoder,
                decoder=decoder,
                critic=None,
            )
            self._cache[FWD_PROVER_KEY] = fwd_prover

        goal_key = f"{prefix}-goals"  # specific key by (split, env_name)
        if goal_key in self._cache:
            forward_goals = self._cache[goal_key]
            logger.info(
                f"Task: {prefix}: found {len(forward_goals)} fwd goals in cache"
            )
        else:
            env_helper = FwdEnvHelper.from_trainer_args(
                dico=self.dico, params=self.params, env=env, env_name=env_name
            )
            goal_factory = env_helper.get_goal_factory()
            forward_goals = goal_factory.build_forward_goals(
                split, debug=self.params.debug.train
            )
            self._cache[goal_key] = forward_goals

        def _goals() -> Iterator[Tuple[int, ForwardGoal]]:
            yield from enumerate(forward_goals)

        logger.info("Starting to prove!")
        outputs = fwd_prover.generate_proofs(_goals())

        n_solved = 0
        total = 0
        n_generated = 0

        proved = []
        failed = []
        for i, proof_search in outputs:
            assert isinstance(proof_search, StandardProofSearch)
            solved = proof_search.info.solved
            name = proof_search.generation.goal.label
            if solved:
                proved.append(name)
            else:
                failed.append(name)
            n_generated += len(proof_search.generation.forward_steps())
            n_solved += int(solved)
            total += 1
        percent_proved = 100 * n_solved / total

        scores[metric_name] = percent_proved
        scores[f"{prefix}-duration"] = time.time() - start
        scores[f"{prefix}-generated"] = n_generated / total
        logger.info(f"{prefix}: proved {n_solved}/{total} ({percent_proved:.3f}%)")
        logger.info(f"{prefix} avg generated: {n_generated / total:.3f}")
        logger.info(f"{prefix} prover stats: {json.dumps(fwd_prover.stats)}")

        if (
            self.params.async_fwd_eval_freq > 0
            and self.trainer.epoch % self.params.async_fwd_eval_freq == 0
        ):
            from evariste.forward.utils.launch_utils import launch_with_submitit
            from evariste.forward.cli.prove import try_prove
            from evariste.forward.cli.prove import Config
            from evariste.forward.forward_model_configs import ModelConfig
            from evariste.forward.forward_model_configs import SAMPLING_PROVER_CFG

            metric_key = f"{prefix}-proven"
            epoch = self.trainer.epoch

            eval_path = Path(self.params.dump_path) / f"{metric_key}" / f"epoch_{epoch}"
            if eval_path.exists():  # was checkpointed at wrong moment ?
                return
            eval_path.mkdir(parents=True)
            checkpoint_path = Path(self.params.dump_path) / f"checkpoint.{epoch}.pth"
            dst_ckpt = eval_path / f"checkpoint.{self.trainer.epoch}.pth"
            shutil.copy(str(checkpoint_path), str(dst_ckpt))

            eval_uuid = f"{self.params.exp_name}.{self.params.exp_id}.{epoch}"

            prove_cfg = SAMPLING_PROVER_CFG
            prove_cfg.async_model = True
            cfg = Config(
                model=ModelConfig(ckpt=str(dst_ckpt), name=eval_uuid),
                prover=prove_cfg,
                split=split,
                slurm=True,
                n_trials_by_trm=128,
                n_jobs=32,
                partition=clusterify_partitions("Theorem_Proving"),
                slurm_mem_gb=80 if env_name == "lean" else 50,
                output_path=str(eval_path),
            )
            try:
                launch_with_submitit(
                    try_prove,
                    cfg=cfg,
                    copy_workdir=False,
                    exp_name=eval_uuid,
                    verbose=False,
                )
            except (FailedJobError, FailedSubmissionError, FileNotFoundError) as e:
                logger.error(f"{type(e)} error: {e} in launch_async")
            else:
                logger.info(f"Launched eval in {eval_path}")
                scores[metric_key] = str(eval_path)

    def evaluate_seq2seq_discriminator(self, scores, split, task):
        if split != "valid":
            # we should use train here, but adversarial training is incompatible
            # with eval on train (because the sender/receiver)
            return

        discriminator = self.modules["discriminator"]
        classifier = self.modules["classifier"]
        discriminator = (
            discriminator.module if self.params.slurm_conf.multi_gpu else discriminator
        )
        classifier = (
            classifier.module if self.params.slurm_conf.multi_gpu else classifier
        )
        discriminator.eval()
        classifier.eval()

        # we use train iterator on purpose, since it is the only one with replay buffer
        iterator = self.trainer.iterators[task]

        n_eval_disc = 500

        predictions = []
        targets = []
        disc_inputs = []
        while len(predictions) < n_eval_disc:
            batch = next(iterator)
            disc_inp = batch["disc_inp"]
            disc_inp_len = batch["disc_inp_len"]
            disc_tgt = batch["disc_tgt"]
            bs = len(disc_tgt)

            disc_inp, disc_inp_len = to_cuda(disc_inp, disc_inp_len)

            # classify
            tensor = discriminator(
                "fwd", causal=False, tokens=disc_inp, lengths=disc_inp_len
            )
            predicted = classifier(tensor[:, 0])
            assert predicted.size() == (bs, 1)
            theses_predictions = torch.sigmoid(predicted.view(-1)).cpu().tolist()
            predictions.extend(theses_predictions)
            targets.extend(disc_tgt.cpu().long().tolist())

            for i in range(len(disc_inp)):
                disc_inputs.append(disc_inp[i, : disc_inp_len[i]].tolist())

        # export generations
        if self.params.slurm_conf.is_master:
            name = f"disc.{split}.{scores['epoch']}"
            dir = Path(self.params.dump_path) / "disc_results"
            dir.mkdir(exist_ok=True)

            fpath = dir / name
            with fpath.open("w") as f:
                for sent, score, tgt in zip(disc_inputs, predictions, targets):
                    words = [self.dico[wid] for wid in sent]
                    sent = " ".join(words)
                    f.write(f"{score:.3f} {tgt} {sent}\n")
            logger.info(f"Exported {len(disc_inputs)} {split} generations to {fpath}")

        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        class_preds = [int(s >= 0.5) for s in predictions]
        acc = accuracy_score(targets, class_preds)
        scores[f"{split}-{task}-disc_acc"] = acc
        prec = precision_score(targets, class_preds)
        scores[f"{split}-{task}-disc_prec"] = prec
        rec = recall_score(targets, class_preds)
        scores[f"{split}-{task}-disc_rec"] = rec
        try:
            auc = roc_auc_score(targets, predictions)
        except ValueError:
            auc = -1
        scores[f"{split}-{task}-disc_auc"] = auc
        scores[f"{split}-{task}-disc_proportion_human"] = sum(targets) / len(targets)

    def evaluate_bwd_prover(
        self, scores, env_name: str, split: str, decoder_type: str = "decoder"
    ):
        """
        Async evaluation with backward prover.
        """
        to_prove_labels = None
        if env_name == "mm":
            dataset: DatasetConf = self.params.mm.dataset
        elif env_name == "hl":
            dataset = self.params.hl.dataset
        elif env_name == "eq":
            dataset = self.params.eq.dataset
            if split in {"valid", "test"}:
                task = (
                    "eq_bwd_graph_seq2seq"
                    if "eq_bwd_graph_seq2seq" in self.params.tasks
                    else "eq_bwd_rwalk_seq2seq"
                )
                to_prove_labels = self.envs[env_name].labels[(task, split)]
            else:
                assert split == "identities"
                to_prove_labels = self.envs[env_name].labels[split]
        elif env_name == "lean":
            dataset = self.params.lean.dataset
        elif env_name in ConfStore:
            dataset = ConfStore[env_name]
        else:
            raise RuntimeError(f"Cannot run bwd_prover for env {env_name}")

        assert decoder_type in ["decoder", "big_decoder"]
        suffix = "" if decoder_type == "decoder" else "-bigdec"
        prefix = f"{split}-{env_name}{suffix}"

        beam_path = Path(
            os.path.join(self.params.dump_path, f"checkpoint.{self.trainer.epoch}.pth")
        )

        # only the master process takes care of evaluation
        if not self.params.slurm_conf.is_master:
            return

        # decoding / prover parameters
        conditioning_kind = ConditioningKind.No
        if any(COND_TOK in task for task in self.params.parsed_tasks(env_name)):
            conditioning_kind = ConditioningKind.GoodOrRandom

        if self.params.eval_greedy:
            # Greedy eval, runs after each epoch.
            logger.info(f"Launching Greedy eval on {env_name} ({split})")

            start = time.time()
            to_prove = get_goals_to_prove(
                dataset,
                split,
                n_to_prove=self.params.n_th_to_prove,
                labels=to_prove_labels,
            )
            logger.info(f"Found {len(to_prove)} goals to prove.")
            torch.cuda.empty_cache()  # hopefully that's enough?

            decoding_params: DecodingParams = ConfStore["decoding_greedy"]
            decoding_params.q_conditioning = self.params.mcts_train.q_conditioning
            decoding_params.q_conditioning_inference_ts = 1.0

            mcts_params: MCTSParams = ConfStore["mcts_fast"]
            mcts_params.no_critic |= not self.params.mcts_train.train_critic
            greedy_dir = (
                Path(self.params.dump_path)
                / "greedy_eval"
                / f"{self.trainer.epoch}"
                / split
            )
            os.makedirs(greedy_dir, exist_ok=True)
            prover_params = ProverParams(
                n_simultaneous_proofs=500,
                beam_path=beam_path,
                dump_path=greedy_dir,
                mcts=mcts_params,
                prover_kind=ProverKind.BackwardGreedy,
                beam_kind=BeamSearchKind.Manual,
                conditioning_kind=conditioning_kind,
                no_proof_check=True,
                quiet=True,
                print_status=False,
            )

            # env specific decoding / proving parameters
            if isinstance(dataset, LeanDatasetConf):
                dataset.lean_cluster = False

            # run prover
            try:
                proofs = bwd_prove(
                    dataset,
                    decoding=decoding_params,
                    prover_params=prover_params,
                    to_prove=to_prove,
                    decoder_type=decoder_type,
                )
            except DeadLean as e:
                proofs = []
                logger.warning(f"Greedy prover raised DeadLean: {e}")

            # stats on proven
            proved_ratio = len(proofs) / len(to_prove)
            scores[f"{prefix}-proven-greedy"] = proved_ratio
            logger.info(f"Greedy {suffix} eval took {time.time() - start}s")
            logger.info(
                f"Proved {len(proofs)} / {len(to_prove)} ({100 * proved_ratio:.2f}%) goals"
            )
            logger.info(
                f"__PROVED_THEOREMS__: {', '.join([p.goal.label for p in proofs])}"
            )

            # stats on proof sizes / lengths
            proof_sizes = [get_proof_size(p.proof) for p in proofs]
            proof_depths = [get_proof_depth(p.proof) for p in proofs]
            mean_size = float(np.mean(proof_sizes)) if len(proofs) > 0 else -1
            mean_depth = float(np.mean(proof_depths)) if len(proofs) > 0 else -1
            scores[f"{prefix}-proven-greedy-mean-size"] = mean_size
            scores[f"{prefix}-proven-greedy-mean-depth"] = mean_depth
            logger.info(f"Mean proof size: {mean_size}")
            logger.info(f"Mean proof depth: {mean_depth}")
            for k in [25, 50, 75]:
                p_size = float(np.percentile(proof_sizes, k)) if len(proofs) > 0 else -1
                p_depth = (
                    float(np.percentile(proof_depths, k)) if len(proofs) > 0 else -1
                )
                # scores[f"{prefix}-proven-greedy-size-percentile{k}"] = p_size
                # scores[f"{prefix}-proven-greedy-depth-percentile{k}"] = p_depth
                logger.info(f"Proof size percentile@{k}: {p_size}")
                logger.info(f"Proof depth percentile@{k}: {p_depth}")

        # optionally run the async eval once in a while
        if (
            self.params.async_bwd_eval_freq > 0
            and self.trainer.epoch % self.params.async_bwd_eval_freq == 0
            and (self.trainer.epoch > 0 or not self.params.async_bwd_eval_skip_zero)
            and not self.params.debug.debug
        ):
            logger.info("Launching Async eval")
            root_dir = get_path_bwd_proving_eval(
                root=self.params.dump_path,
                lang=env_name,
                split=split,
                epoch=self.trainer.epoch,
                decoder_type=decoder_type,
            )
            scores[f"{prefix}-proven"] = root_dir

            # use same amount of GPU for eval as we do for training, but no less than 4
            n_machines = max(min(self.params.slurm_conf.world_size, 16), 4)

            mcts_params = ConfStore["mcts_fast"]
            mcts_params.no_critic |= not self.params.mcts_train.train_critic
            decoding_params = self.params.async_bwd_eval_dec_params
            decoding_params.q_conditioning = self.params.mcts_train.q_conditioning
            decoding_params.q_conditioning_inference_ts = 1.0

            # decoding / prover parameters
            conditioning_kind = ConditioningKind.No
            if any(COND_TOK in task for task in self.params.parsed_tasks(env_name)):
                conditioning_kind = ConditioningKind.GoodOrRandom

            zmq_prover_params = ZMQProverParams(
                prover=ProverParams(
                    n_simultaneous_proofs=5,
                    beam_path=beam_path,
                    mcts=mcts_params,
                    prover_kind=ProverKind.BackwardMCTS,
                    beam_kind=BeamSearchKind.Manual,
                    conditioning_kind=conditioning_kind,
                    dump_path=Path(root_dir),
                    quiet=True,
                ),
                decoding=decoding_params,
                n_machines=n_machines,
                max_attempts=self.params.async_bwd_eval_max_attempts,
                partition=clusterify_partitions("Theorem_Proving"),
                root_dir=Path(root_dir),
                n_th_to_prove=self.params.n_th_to_prove,
                shuffle_seed=43,
                decoder_type=decoder_type,
                dump_proofs=True,
                max_restarts=0,  # always 0 for these to avoid fork bombing the cluster if something goes wrong.
            )
            zmq_prover_params.set_dataset(env_name, dataset)

            try:
                job = launch_async(
                    zmq_prover_params,
                    split=split,
                    name=f"{self.params.exp_name}_e{self.trainer.epoch}",
                    timeout_min=self.params.async_bwd_eval_timeout,
                )
                job.cancel_at_deletion()
                self._running_eval_jobs[
                    (env_name, split, self.trainer.epoch, decoder_type)
                ] = job

                logger.info(
                    f"{self.params.slurm_conf.global_rank} Launched ZMQ prover job on "
                    f"{n_machines} machines on partition {zmq_prover_params.partition}: "
                    f"{job.job_id} for {self.trainer.epoch}/{split}/{env_name}"
                )
            except (FailedJobError, FailedSubmissionError, FileNotFoundError) as e:
                logger.error(f"{type(e)} error: {e} in launch_async")

    def close(self):
        logger.info("Closing Evaluator ...")
        if FWD_PROVER_KEY in self._cache:
            # probably not needed since we don't use another process
            # in forward prover, but better be safe
            from evariste.forward.forward_prover import ForwardProver

            fwd_prover = self._cache[FWD_PROVER_KEY]
            assert isinstance(fwd_prover, ForwardProver)
            logger.info("Closing fwd_prover in evaluator")
            fwd_prover.close()
            logger.info("Closed fwd_prover in evaluator")
            self._cache.pop(FWD_PROVER_KEY)
        logger.info("Closed Evaluator")


def convert_to_text(batch, lengths, dico, eos_index):
    """
    Convert a batch of sentences to a list of text sentences.
    """
    batch = batch.cpu()
    lengths = lengths.cpu()

    bs, slen = batch.shape
    assert lengths.max() == slen and lengths.shape == (bs,)
    assert (batch[:, 0] == eos_index).sum() == bs
    assert (batch == eos_index).sum() == 2 * bs
    sentences = []
    for i in range(bs):
        assert batch[i, 0] == eos_index
        assert batch[i, lengths[i] - 1] == eos_index
        word_ids = batch[i, 1 : lengths[i] - 1].tolist()
        sentences.append(" ".join(dico[j] for j in word_ids))
    return sentences


def eval_moses_bleu(hyp, ref):
    """
    Given a file of hypothesis and reference files,
    evaluate the BLEU score using Moses scripts.
    """
    assert os.path.isfile(hyp)
    assert os.path.isfile(ref) or os.path.isfile(ref + "0")
    assert os.path.isfile(BLEU_SCRIPT_PATH)
    command = f"{BLEU_SCRIPT_PATH} {ref} < {hyp}"
    p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    result = p.communicate()[0].decode("utf-8")
    if result.startswith("BLEU"):
        return float(result[7 : result.index(",")])
    else:
        logger.warning(f'Impossible to parse BLEU score: "{result}"')
        return -1
