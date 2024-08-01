# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch

from params import ConfStore
from evariste.model.modules import reload_model
from evariste.model.data.dictionary import Dictionary
from evariste.model.transformer import TransformerModel
from evariste.model.transformer_args import ModelArgs


def batch_sentences(x, dico):
    """
        Create a batch of padded sequences.
        """
    assert type(x) is list
    assert all(type(xs) is list for xs in x)

    # sequence lengths
    x = [[dico.word2id[w] for w in s] for s in x]
    bs = len(x)
    xlen = torch.LongTensor([len(s) for s in x])

    # merge sequences into a batch
    x_batch = torch.full((bs, max(xlen)), dico.pad_index, dtype=torch.long)
    for sid, (xl, xs) in enumerate(zip(xlen, x)):
        assert len(xs) == xl
        x_batch[sid, :xl] = torch.LongTensor(xs)

    return x_batch, xlen


class TestReloadModel:
    def test_reload_same_dico(self):
        """
        Test the reloading of a src model into a tgt model
        when dictionary are the same.
        Need to test equality of output tensor, logits and loss
        """
        torch.manual_seed(0)
        vocab = {"a": 10, "b": 8, "c": 7, "d": 2}
        dico = Dictionary.create_empty()
        dico.add_vocab(vocab)
        model_args: ModelArgs = ConfStore["model_default"]
        model_args.fp16 = False
        model_args._check_and_mutate_args()

        tgt_model = TransformerModel(
            model_args, dico, is_encoder=True, with_output=True
        )
        src_model = TransformerModel(
            model_args, dico, is_encoder=True, with_output=True
        )
        tgt_model.eval()
        src_model.eval()

        # create inputs
        x = [["a", "b", "d"]]
        x, xlen = batch_sentences(x, dico)
        pred_mask = torch.ones(x.size(), dtype=torch.bool)

        # test before reloading not equal
        tgt_tensor = tgt_model("fwd", causal=False, tokens=x, lengths=xlen)
        tgt_logits, tgt_loss = tgt_model(
            "compute_loss", tensor=tgt_tensor, pred_mask=pred_mask, target=x[0]
        )
        src_tensor = src_model("fwd", causal=False, tokens=x, lengths=xlen)
        src_logits, src_loss = src_model(
            "compute_loss", tensor=src_tensor, pred_mask=pred_mask, target=x[0]
        )
        assert not (tgt_tensor - src_tensor).abs().max().item() < 1e-6
        assert not (tgt_logits - src_logits).abs().max().item() < 1e-6
        assert not (tgt_loss - src_loss).abs().max().item() < 1e-6

        # reload src into tgt and test equal
        src = {"model": src_model.state_dict(), "dico": dico}
        reload_model(tgt_model, dico, src, "encoder")
        tgt_tensor = tgt_model("fwd", causal=False, tokens=x, lengths=xlen)
        tgt_logits, tgt_loss = tgt_model(
            "compute_loss", tensor=tgt_tensor, pred_mask=pred_mask, target=x[0]
        )
        assert (tgt_tensor - src_tensor).abs().max().item() < 1e-6
        assert (tgt_logits - src_logits).abs().max().item() < 1e-6
        assert (tgt_loss - src_loss).abs().max().item() < 1e-6

    def test_reload_same_dico_diff_order(self):
        """
        Test the reloading of a src model into a tgt model
        when dictionary have the same vocabulary but not same id.
        Useful to test reload of pred_layer.
        Need to test equality of output tensor, logits and loss
        """
        torch.manual_seed(0)
        src_vocab = {"a": 10, "b": 8, "c": 7, "d": 2}
        src_dico = Dictionary.create_empty()
        src_dico.add_vocab(src_vocab)

        tgt_vocab = {"a": 1, "b": 8, "c": 2, "d": 6}
        tgt_dico = Dictionary.create_empty()
        tgt_dico.add_vocab(tgt_vocab)

        assert tgt_dico != src_dico
        assert all(k in tgt_dico and k in src_dico for k in ["a", "b", "c", "d"])
        assert not all(
            tgt_dico.word2id[k] == src_dico.word2id[k] for k in ["a", "b", "c", "d"]
        )

        model_args: ModelArgs = ConfStore["model_default"]
        model_args.fp16 = False
        model_args._check_and_mutate_args()

        tgt_model = TransformerModel(
            model_args, tgt_dico, is_encoder=True, with_output=True
        )
        src_model = TransformerModel(
            model_args, src_dico, is_encoder=True, with_output=True
        )
        tgt_model.eval()
        src_model.eval()

        # create inputs
        x = [["a", "b", "d"]]
        tgt_x, tgt_xlen = batch_sentences(x, tgt_dico)
        tgt_pred_mask = torch.ones(tgt_x.size(), dtype=torch.bool)
        src_x, src_xlen = batch_sentences(x, src_dico)
        src_pred_mask = torch.ones(src_x.size(), dtype=torch.bool)

        # test before reloading not equal
        tgt_tensor = tgt_model("fwd", causal=False, tokens=tgt_x, lengths=tgt_xlen)
        tgt_logits, tgt_loss = tgt_model(
            "compute_loss", tensor=tgt_tensor, pred_mask=tgt_pred_mask, target=tgt_x[0]
        )
        src_tensor = src_model("fwd", causal=False, tokens=src_x, lengths=src_xlen)
        src_logits, src_loss = src_model(
            "compute_loss", tensor=src_tensor, pred_mask=src_pred_mask, target=src_x[0]
        )
        assert not (tgt_tensor - src_tensor).abs().max().item() < 1e-6
        assert not (tgt_loss - src_loss).abs().max().item() < 1e-6

        # reload and test equality for tensor only - logits are not equal when not same dico
        src = {"model": src_model.state_dict(), "dico": src_dico}
        reload_model(tgt_model, tgt_dico, src, "encoder")
        tgt_tensor = tgt_model("fwd", causal=False, tokens=tgt_x, lengths=tgt_xlen)
        tgt_logits, tgt_loss = tgt_model(
            "compute_loss", tensor=tgt_tensor, pred_mask=tgt_pred_mask, target=tgt_x[0]
        )
        assert (tgt_tensor - src_tensor).abs().max().item() < 1e-6
        assert (tgt_loss - src_loss).abs().max().item() < 1e-6

    def test_reload_diff_dico(self):
        """
        Test the reloading of a src model into a tgt model
        when dictionary are different.
        Need to test equality of output tensor.
        Here logits and loss cannot be compared.
        """
        torch.manual_seed(0)
        src_vocab = {"a": 10, "b": 8, "c": 7, "d": 2}
        src_dico = Dictionary.create_empty()
        src_dico.add_vocab(src_vocab)

        tgt_vocab = {"a": 10, "b": 8, "zz": 4, "xx": 3, "d": 2}
        tgt_dico = Dictionary.create_empty()
        tgt_dico.add_vocab(tgt_vocab)

        assert tgt_dico != src_dico
        assert all(k in tgt_dico and k in src_dico for k in ["a", "b", "d"])
        assert not all(
            tgt_dico.word2id[k] == src_dico.word2id[k] for k in ["a", "b", "d"]
        )

        model_args: ModelArgs = ConfStore["model_default"]
        model_args.fp16 = False
        model_args._check_and_mutate_args()

        tgt_model = TransformerModel(
            model_args, tgt_dico, is_encoder=True, with_output=True
        )
        src_model = TransformerModel(
            model_args, src_dico, is_encoder=True, with_output=True
        )
        tgt_model.eval()
        src_model.eval()

        # create inputs
        x = [["a", "b", "d"]]
        tgt_x, tgt_xlen = batch_sentences(x, tgt_dico)
        tgt_pred_mask = torch.ones(tgt_x.size(), dtype=torch.bool)
        src_x, src_xlen = batch_sentences(x, src_dico)
        src_pred_mask = torch.ones(src_x.size(), dtype=torch.bool)

        # test before reloading not equal
        tgt_tensor = tgt_model("fwd", causal=False, tokens=tgt_x, lengths=tgt_xlen)
        tgt_logits, tgt_loss = tgt_model(
            "compute_loss", tensor=tgt_tensor, pred_mask=tgt_pred_mask, target=tgt_x[0]
        )
        src_tensor = src_model("fwd", causal=False, tokens=src_x, lengths=src_xlen)
        assert not (tgt_tensor - src_tensor).abs().max().item() < 1e-6

        # reload and test equality for tensor only - logits are not equal when not same dico
        src = {"model": src_model.state_dict(), "dico": src_dico}
        reload_model(tgt_model, tgt_dico, src, "encoder")

        tgt_tensor = tgt_model("fwd", causal=False, tokens=tgt_x, lengths=tgt_xlen)
        assert (tgt_tensor - src_tensor).abs().max().item() < 1e-6
