# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from typing import Optional, Tuple, List, Set, Dict
from collections import defaultdict
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
import os
import hashlib
import numpy as np
import torch

from evariste import json as json
from params import Params
from evariste.refac.utils import safe_load


logger = getLogger()


BOS_WORD = "<s>"
EOS_WORD = "</s>"
PAD_WORD = "<pad>"
UNK_WORD = "<unk>"

N_SPECIAL_WORDS = 50
SPECIAL_WORD = "<SPECIAL_%i>"

MASK_WORD = "<MASK>"  # mask word (masked language modeling)
B_CMD_WORD = "<COMMAND>"  # begin command
E_CMD_WORD = "</COMMAND>"  # end command
B_GOAL_WORD = "<GOAL>"  # begin goal
E_GOAL_WORD = "</GOAL>"  # end goal
B_HYP_WORD = "<HYP>"  # begin hypothesis
M_HYP_WORD = "<HYP_NAME>"  # separate hypothesis name (1 token) from its content
E_HYP_WORD = "</HYP>"  # end hypothesis
EXCEPTION_WORD = "<EXCEPTION>"  # exception word
EMPTY_GOAL_WORD = "<EMPTY_GOAL>"  # empty goal
UNAFFECTED_GOAL_WORD = "<UNAFFECTED_GOAL>"  # unaffected goal
B_SUBST_WORD = "<SUBST>"  # begin substitution
M_SUBST_WORD = "<SUBST_NAME>"  # separate substitution name (1 token) from its content
E_SUBST_WORD = "</SUBST>"  # end substitution
B_STACK_WORD = "<STACK>"  # beginning  of stack
M_STACK_WORD = "<STACK_SEP>"  # stack separator
E_STACK_WORD = "</STACK>"  # end of stack
EMB_WORD = "<EMB>"  # token corresponding to an embedding
EMPTY_NODE_WORD = "<EMPTY_NODE>"  # empty node
B_THEOREM_WORD = "<THEOREM>"  # begin theorem
E_THEOREM_WORD = "</THEOREM>"  # end theorem
EOU_WORD = "<END_OF_USEFUL>"  # end of useful word
SOLVABLE_WORD = "<SOLVABLE>"
NON_SOLVABLE_WORD = "<NON_SOLVABLE>"
CRITIC_WORD = "<CRITIC>"
B_NODE_WORD = "<NODE>"  # start of node
E_NODE_WORD = "</NODE>"  # end of node
B_HEAD_WORD = "<HEAD>"  # start of head node
E_HEAD_WORD = "</HEAD>"  # end of head node
STOP_WORD = "<STOP>"
NEWLINE_WORD = "<NEWLINE>"
REPEAT_GOAL_WORD = "<REPEAT_GOAL>"
GOAL_STATEMENT_WORD = "<GOAL_STATEMENT>"

# Used to predict the effect of a tactic
SUCCESS_WORD = "<SUCCESS>"
NO_EFFECT_WORD = "<NO_EFFECT>"
UNK_ERROR_WORD = "UNK_ERROR"

# Used to predict if a generation will be proved or not
PROVED_WORD = "<PROVED>"
UNPROVED_WORD = "<UNPROVED>"

# used by cclm and gan
GEN_REAL = "<GEN_REAL>"
GEN_FAKE = "<GEN_FAKE>"

# used to provide context on open namespaces in lean
B_NS_WORD = "<NS>"
E_NS_WORD = "</NS>"

# used to do simultaneous training of fwd and bwd
BWD_WORD = "<BWD>"

# used to do condition the forward
TARGET_IS_GOAL_WORD = "<TARGET_IS_GOAL>"

# used to create a fake filling tactic, so that tactic probabilities sum to 1,
# and error tactics, for tactics that generated an error
TACTIC_FILL_WORD = "<TACTIC_FILL>"
TACTIC_PARSE_ERROR_WORD = "<TACTIC_PARSE_ERROR>"
TACTIC_ENV_ERROR_WORD = "<TACTIC_ENV_ERROR>"


SQUID_TOK = "🦑"


SPECIAL_WORDS = [
    MASK_WORD,
    B_CMD_WORD,
    E_CMD_WORD,
    B_GOAL_WORD,
    E_GOAL_WORD,
    B_HYP_WORD,
    M_HYP_WORD,
    E_HYP_WORD,
    EXCEPTION_WORD,
    EMPTY_GOAL_WORD,
    UNAFFECTED_GOAL_WORD,
    B_SUBST_WORD,
    M_SUBST_WORD,
    E_SUBST_WORD,
    B_STACK_WORD,
    M_STACK_WORD,
    E_STACK_WORD,
    EMB_WORD,
    EMPTY_NODE_WORD,
    B_THEOREM_WORD,
    E_THEOREM_WORD,
    EOU_WORD,
    SOLVABLE_WORD,
    NON_SOLVABLE_WORD,
    CRITIC_WORD,
    B_NODE_WORD,
    E_NODE_WORD,
    B_HEAD_WORD,
    E_HEAD_WORD,
    STOP_WORD,
    SUCCESS_WORD,
    NO_EFFECT_WORD,
    UNK_ERROR_WORD,
    PROVED_WORD,
    UNPROVED_WORD,
    GEN_REAL,
    GEN_FAKE,
    B_NS_WORD,
    E_NS_WORD,
    BWD_WORD,
    TARGET_IS_GOAL_WORD,
    NEWLINE_WORD,
    REPEAT_GOAL_WORD,
    GOAL_STATEMENT_WORD,
    TACTIC_FILL_WORD,
    TACTIC_PARSE_ERROR_WORD,
    TACTIC_ENV_ERROR_WORD,
]
SPECIAL_WORDS = SPECIAL_WORDS + [
    SPECIAL_WORD % i for i in range(len(SPECIAL_WORDS), N_SPECIAL_WORDS)
]


@dataclass
class DicoConf(Params):
    n_words: int = -1
    bos_index: int = -1
    eos_index: int = -1
    pad_index: int = -1
    unk_index: int = -1
    emb_index: int = -1
    mask_index: int = -1


class Dictionary(object):
    def __init__(
        self,
        id2word: Dict[int, str],
        word2id: Dict[str, int],
        counts: Dict[str, int],
        frozen: bool = False,
    ):
        assert len(id2word) == len(word2id) == len(counts)
        self.id2word = id2word
        self.word2id = word2id
        self.counts = counts
        self.update_special_words()
        self.update_words_with_spaces()
        self.bos_word = BOS_WORD
        self.eos_word = EOS_WORD
        self.pad_word = PAD_WORD
        self.unk_word = UNK_WORD
        self.bos_index = word2id[BOS_WORD]
        self.eos_index = word2id[EOS_WORD]
        self.pad_index = word2id[PAD_WORD]
        self.unk_index = word2id[UNK_WORD]
        self.critic_index = word2id[CRITIC_WORD]
        if EMB_WORD in word2id:
            self.emb_index = word2id.get(EMB_WORD)
        self.stop_word = STOP_WORD
        self.check_valid()
        logger.info(
            f"Created dictionary with {len(id2word)} words ({sum(counts.values())} total)."
        )
        self.frozen = frozen
        self.unk_seen: Dict[str, int] = defaultdict(int)
        self.total_unk = 0

    def __len__(self):
        """
        Returns the number of words in the dictionary.
        """
        return len(self.id2word)

    def __getitem__(self, i):
        """
        Returns the word of the specified index.
        """
        return self.id2word[i]

    def __contains__(self, w):
        """
        Returns whether a word is in the dictionary.
        """
        return w in self.word2id

    def __eq__(self, y):
        """
        Compare this dictionary with another one.
        """
        assert isinstance(y, Dictionary), y
        self.check_valid()
        y.check_valid()
        if len(self.id2word) != len(y):
            return False
        return all(self.id2word[i] == y[i] for i in range(len(y)))

    @property
    def conf(self) -> DicoConf:
        return DicoConf(
            n_words=len(self),
            bos_index=self.index(BOS_WORD),
            eos_index=self.index(EOS_WORD),
            pad_index=self.index(PAD_WORD),
            unk_index=self.index(UNK_WORD),
            emb_index=self.index(EMB_WORD),
            mask_index=self.index(MASK_WORD),
        )

    def update_special_words(self):
        """
        Update potential new special words.
        """
        for i, new_w in enumerate(SPECIAL_WORDS):
            old_w = self.id2word[4 + i]
            if old_w == new_w:
                continue
            logger.warning(f"Updating special word: {old_w} -> {new_w}")
            self.id2word[4 + i] = new_w
            self.word2id[new_w] = 4 + i
            self.counts[new_w] = self.counts[old_w]
            del self.word2id[old_w]
            del self.counts[old_w]

    def update_words_with_spaces(self):
        """
        Small hack to remap a few incorrect words with accidental spaces inside.
        """
        for w_old, w_new in [("APPLY_INSTANCE FAILED", "APPLY_INSTANCE_FAILED")]:
            if w_old not in self.word2id:
                continue
            logger.warning(f"Found {w_old} in the dictionary. Remapping to {w_new}")
            assert w_new not in self.word2id
            idx = self.word2id.pop(w_old)
            self.word2id[w_new] = idx
            self.counts[w_new] = self.counts.pop(w_old)
            self.id2word[idx] = w_new

    def check_valid(self, sorted_counts: bool = False):
        """
        Check that the dictionary is valid.
        """
        assert self.bos_index == 0
        assert self.eos_index == 1
        assert self.pad_index == 2
        assert self.unk_index == 3
        # assert all(self.id2word[4 + i] == w for i, w in enumerate(SPECIAL_WORDS))
        assert len(self.id2word) == len(self.word2id) == len(self.counts)
        assert set(self.word2id.keys()) == set(self.counts.keys())
        assert all(self.word2id[self.id2word[i]] == i for i in range(len(self.id2word)))
        # assert all(
        #     self.counts[self.id2word[i]] == 0 for i in range(4 + N_SPECIAL_WORDS)
        # ), {
        #     self.id2word[i]
        #     for i in range(4 + N_SPECIAL_WORDS)
        #     if self.counts[self.id2word[i]] > 0
        # }
        if sorted_counts:
            last_count = 1e18
            for i in range(4 + N_SPECIAL_WORDS, len(self.id2word)):
                count = self.counts[self.id2word[i]]
                assert count <= last_count
                last_count = count
        for k in self.word2id.keys():
            assert " " not in k, k

    def index(self, word: str):
        """
        Returns the index of the specified word.
        """
        try:
            return self.word2id[word]
        except KeyError:
            self.unk_seen[word] += 1
            self.total_unk += 1
            logger.error(f"[Dictionary] {self.total_unk} UNK")
            logger.error(f"[Dictionary] {json.dumps(self.unk_seen)} UNK")
            raise

    def max_vocab(self, max_vocab: int):
        """
        Limit the vocabulary size.
        """
        assert max_vocab >= 1
        init_size = len(self)
        self.id2word = {k: v for k, v in self.id2word.items() if k < max_vocab}
        self.word2id = {v: k for k, v in self.id2word.items()}
        self.counts = {k: v for k, v in self.counts.items() if k in self.word2id}
        self.check_valid()
        logger.info(
            f"Maximum vocabulary size: {max_vocab}. Dictionary size: "
            f"{init_size} -> {len(self)} (removed {init_size - len(self)} words)."
        )

    def min_count(self, min_count: int):
        """
        Threshold on the word frequency counts.
        """
        assert min_count >= 0
        init_size = len(self)
        self.id2word = {
            k: v
            for k, v in self.id2word.items()
            if self.counts[self.id2word[k]] >= min_count or k < 4 + N_SPECIAL_WORDS
        }
        self.word2id = {v: k for k, v in self.id2word.items()}
        self.counts = {k: v for k, v in self.counts.items() if k in self.word2id}
        self.check_valid()
        logger.info(
            "Minimum frequency count: %i. Dictionary size: %i -> %i (removed %i words)."
            % (min_count, init_size, len(self), init_size - len(self))
        )

    def add_vocab(self, vocab: Dict[str, int]) -> None:
        """
        Merge dictionary with new vocabulary.
        """
        assert not any(" " in k for k in vocab.keys())
        if self.frozen:
            new_words = sorted(vocab.keys() - self.word2id.keys())
            if len(new_words) > 0:
                logger.warning(
                    f"Found {len(new_words)} words that cannot be added be added because "
                    f"the dictionary is frozen: {', '.join(new_words[:100])}"
                )
            return
        n_uni = len(self)
        n_tot = sum(self.counts.values())
        logger.info(
            f"Updating dictionary. Currently {n_uni} words, ({n_tot} total). "
            f"Adding vocabulary with {len(vocab)} words, ({sum(vocab.values())} total) ..."
        )
        # sort vocabulary by token frequency
        vocab_sorted = sorted(vocab.items(), key=lambda x: (-x[1], x[0]))
        for word, count in vocab_sorted:
            if word in self:
                self.counts[word] += count
            else:
                new_id = len(self)
                assert new_id not in self.id2word
                self.id2word[new_id] = word
                self.word2id[word] = new_id
                self.counts[word] = count
        logger.info(
            f"Added {len(self) - n_uni} unique words. "
            f"Now {len(self)} words, ({sum(self.counts.values())} total)."
        )
        self.check_valid()

    def __hash__(self) -> int:
        vocab = ", ".join(self.word2id.keys())
        return int(hashlib.sha256(vocab.encode("utf-8")).hexdigest(), 16)

    @staticmethod
    def create_empty() -> "Dictionary":
        """
        Create an empty dictionary.
        """
        return Dictionary.create_from_vocab_counts({})

    @staticmethod
    def create_from_checkpoint(path) -> "Dictionary":
        """
        Create a dictionary from a checkpoint.
        """
        logger.info(f"Initializing dictionary from checkpoint {path}")
        assert os.path.isfile(path), path
        reloaded = safe_load(path, map_location="cpu")
        logger.info(f"Found {len(reloaded['dico_id2word'])} words in {path}")
        return Dictionary(
            reloaded["dico_id2word"],
            reloaded["dico_word2id"],
            reloaded["dico_counts"],
            frozen=True,
        )

    @staticmethod
    def create_from_pretrained(to_reload: Dict[str, Path]) -> "Dictionary":
        """
        Create a dictionary from pretrained models.
        If multiple paths are provided, all dictionaries should be identical.
        """
        reload_paths: Set[Path] = set(to_reload.values())
        assert len(reload_paths) >= 1
        assert all(path.is_file() for path in reload_paths)
        logger.info(
            f"Initializing dictionary from pretrained models: "
            f"{', '.join(str(path) for path in reload_paths)}"
        )

        dico: Optional[Dictionary] = None
        for i, path in enumerate(reload_paths):
            reloaded = safe_load(path, map_location="cpu")
            _dico = Dictionary(
                reloaded["dico_id2word"],
                reloaded["dico_word2id"],
                reloaded["dico_counts"],
            )
            logger.info(f"Found {len(reloaded['dico_id2word'])} words in {path}")
            if i == 0:
                dico = _dico
            else:
                assert _dico == dico
        assert dico is not None
        return dico

    @staticmethod
    def create_from_vocab_counts(vocab: Dict[str, int]) -> "Dictionary":
        """
        Create a dictionary from a dictionary of words with counts.
        """
        word2id = {BOS_WORD: 0, EOS_WORD: 1, PAD_WORD: 2, UNK_WORD: 3}
        for i, w in enumerate(SPECIAL_WORDS):
            word2id[w] = 4 + i
        counts: Dict[str, int] = {k: 0 for k in word2id.keys()}
        for w, c in sorted(vocab.items(), key=lambda x: (-x[1], x[0])):
            assert w not in word2id and type(c) is int
            word2id[w] = len(word2id)
            counts[w] = c
        id2word: Dict[int, str] = {v: k for k, v in word2id.items()}
        return Dictionary(id2word, word2id, counts, frozen=False)

    @staticmethod
    def create_from_vocab_path(vocab_path: str) -> "Dictionary":
        """
        Create a dictionary from a vocabulary file.
        """
        skipped = 0
        assert os.path.isfile(vocab_path), vocab_path
        word2id = {BOS_WORD: 0, EOS_WORD: 1, PAD_WORD: 2, UNK_WORD: 3}
        for i, w in enumerate(SPECIAL_WORDS):
            word2id[w] = 4 + i
        counts: Dict[str, int] = {k: 0 for k in word2id.keys()}
        f = open(vocab_path, "r", encoding="utf-8")
        for i, line in enumerate(f):
            if "\u2028" in line:
                skipped += 1
                continue
            toks = line.rstrip().split()
            if len(toks) != 2:
                skipped += 1
                continue
            assert len(toks) == 2, (i, toks)
            # assert line[0] not in word2id and line[1].isdigit(), (i, line)
            assert toks[1].isdigit(), (i, toks)
            if toks[0] in word2id:
                skipped += 1
                print("%s already in vocab" % toks[0])
                continue
            if not toks[1].isdigit():
                skipped += 1
                print("Empty word at line %s with count %s" % (i, toks))
                continue
            word2id[toks[0]] = (
                4 + N_SPECIAL_WORDS + i - skipped
            )  # shift because of extra words
            counts[toks[0]] = int(toks[1])
        f.close()
        id2word: Dict[int, str] = {v: k for k, v in word2id.items()}
        dico = Dictionary(id2word, word2id, counts)
        logger.info("Read %i words from the vocabulary file." % len(dico))
        if skipped > 0:
            logger.warning("Skipped %i empty lines!" % skipped)
        return dico

    @staticmethod
    def index_data(path: str, bin_path: Optional[str], dico: "Dictionary") -> Dict:
        """
        Index sentences with a dictionary.
        """
        if bin_path is not None and os.path.isfile(bin_path):
            print("Loading data from %s ..." % bin_path)
            data = safe_load(bin_path, map_location="cpu")
            assert dico == Dictionary(
                data["dico_id2word"], data["dico_word2id"], data["dico_counts"]
            )
            return data

        positions: List[Tuple[int, int]] = []
        sentences: List[int] = []
        unk_words: Dict[str, int] = {}

        # index sentences
        f = open(path, "r", encoding="utf-8")
        for i, line in enumerate(f):
            if i % 1_000_000 == 0 and i > 0:
                print(i)
            s = line.rstrip().split()
            # skip empty sentences
            if len(s) == 0:
                print("Empty sentence in line %i." % i)
            # index sentence words
            count_unk = 0
            indexed = []
            for w in s:
                word_id = dico.index(w)
                # if we find a special word which is not an unknown word, skip the sentence
                if 0 <= word_id < 4 + N_SPECIAL_WORDS and word_id != 3:
                    logger.warning(
                        'Found unexpected special word "%s" (%i)!!' % (w, word_id)
                    )
                    continue
                assert word_id >= 0
                indexed.append(word_id)
                if word_id == dico.unk_index:
                    unk_words[w] = unk_words.get(w, 0) + 1
                    count_unk += 1
            # add sentence
            positions.append((len(sentences), len(sentences) + len(indexed)))
            sentences.extend(indexed)
            sentences.append(1)  # EOS index
        f.close()

        # tensorize data
        positions_arr: np.ndarray = np.array(positions, dtype=np.int64)
        if len(dico) < 1 << 16:
            sentences_arr: np.ndarray = np.array(sentences, dtype=np.uint16)
        elif len(dico) < 1 << 31:
            sentences_arr = np.array(sentences, dtype=np.int32)
        else:
            raise Exception("Dictionary is too big.")
        assert sentences_arr.min() >= 0
        data = {
            "dico_id2word": dico.id2word,
            "dico_word2id": dico.word2id,
            "dico_counts": dico.counts,
            "positions": positions_arr,
            "sentences": sentences_arr,
            "unk_words": unk_words,
        }
        if bin_path is not None:
            print("Saving the data to %s ..." % bin_path)
            torch.save(data, bin_path, pickle_protocol=4)

        return data
