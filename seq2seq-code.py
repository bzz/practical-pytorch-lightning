#!/usr/bin/env python3
# coding=utf-8
"""
PyTorch-lightning (PTL) implementation of seq2seq architecture for
function name suggestions.
"""
import glob
import logging
import math
import os
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import rnn
from torch.utils.data import DataLoader, Dataset

import metrics
from data_generators import text_encoder
from data_generators.text_encoder import SubwordTextEncoder, TextEncoder

SEED = 2334
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

logging.basicConfig(level=logging.INFO)

# TODO(bzz): generate from the problem name
vocab_filepath = "vocab.programming_function_name.8192.subwords"


# Data
class CodeSearchNetRAM(Dataset):
    """Map-style dataset that stores CodeSearchNet data in memory.

       This implementation cuts off source/target encoded seq length hard,
       instead of dropping such examples alltogether, before sampling.
    """
    cut = 400
    split_files = {
        "train": os.path.join("final", "jsonl", "train"),
        "valid": os.path.join("final", "jsonl", "valid"),
        "test": os.path.join("final", "jsonl", "valid"),
    }

    def __init__(self, base: str, enc: TextEncoder, split: str):
        super().__init__()
        self.enc = enc
        self.pd = pd

        split_path = base / Path(self.split_files[split])
        files = sorted(split_path.glob('**/*.gz'))
        print(f'Total number of files: {len(files):,}')
        assert len(files) != 0, "could not find files under %s" % split_path

        # columns_long_list = [  # for debuging
        #     'repo', 'path', 'url', 'code', 'code_tokens', 'func_name',
        #     'language', 'partition'
        # ]
        columns_list = ['code', 'func_name']

        start = time.time()
        self.pd = self._jsonl_list_to_dataframe(files, columns_list)
        print("Loading data took {:.2f}s".format(time.time() - start))

    @staticmethod
    def _jsonl_list_to_dataframe(file_list: List[Path],
                                 columns: List[str]) -> pd.DataFrame:
        """Load a list of jsonl.gz files into a pandas DataFrame."""
        # yapf: disable
        return pd.concat([pd.read_json(f,
                                    orient='records',
                                    compression='gzip',
                                    lines=True)[columns]
                        for f in file_list], sort=False)
        # yapf: enable

    def __getitem__(self, idx: int):
        row = self.pd.iloc[idx]

        # cut, drop class name
        fn_name = row["func_name"][:self.cut]
        fn_name = fn_name.split('.')[-1]  # drop the class name
        fn_name_enc = self.enc.encode(fn_name) + [text_encoder.EOS_ID]

        # cut, drop fn signature
        code = row["code"][:self.cut]
        fn_body = code[code.find("{") + 1:code.find("}")].lstrip().rstrip()
        fn_body_enc = self.enc.encode(fn_body) + [text_encoder.EOS_ID]

        # fn_code_enc_p = fn_code[:20].replace("\n", "\\n")
        # print(f"name:{fn_name}, code:{fn_code_enc_p}")

        return (torch.LongTensor(fn_body_enc), torch.LongTensor(fn_name_enc))

    def __len__(self):
        return len(self.pd)


def pack_collate(batch):
    """Packs variable-length input, target and decoder input to PackedSequence"""

    def pack_batch_first(
            sequence) -> Tuple[rnn.PackedSequence, torch.Tensor, List[int]]:
        """pack_padded_sequence with batch_first=True"""
        lengths = [v.size(0) for v in sequence]
        padded = rnn.pad_sequence(sequence, batch_first=True)
        return (rnn.pack_padded_sequence(padded,
                                         lengths,
                                         batch_first=True,
                                         enforce_sorted=False), padded, lengths)

    def shift_right(t: torch.Tensor) -> torch.Tensor:
        st = torch.roll(t, 1, 0)
        st[0] = text_encoder.BOS_ID
        return st

    (fns_code, fns_names) = zip(*batch)
    inps_pck, _, _ = pack_batch_first(fns_code)
    tgts_pck, _tgts_pad, _tgts_len = pack_batch_first(fns_names)

    _tgts_shifted_pad = torch.stack([shift_right(t) for t in _tgts_pad])
    tgts_shifted_pck = rnn.pack_padded_sequence(_tgts_shifted_pad,
                                                _tgts_len,
                                                batch_first=True,
                                                enforce_sorted=False)
    return inps_pck, tgts_pck, tgts_shifted_pck


def pad_collate(batch):
    """Padds input and target to the same length"""
    (fn_code, fn_name) = zip(*batch)

    inps_pad = rnn.pad_sequence(fn_code,
                                batch_first=True,
                                padding_value=text_encoder.PAD_ID)
    tgts_pad = rnn.pad_sequence(fn_name,
                                batch_first=True,
                                padding_value=text_encoder.PAD_ID)

    return torch.utils.data.dataloader.default_collate(
        list(zip(*(inps_pad, tgts_pad))))


def elementwise_apply(fn, *args):
    """A hack to apply fn like nn.Embedding, F.log_softmax to PackedSequence"""
    return rnn.PackedSequence(
        fn(*[(arg.data if type(arg) == rnn.PackedSequence else arg)
             for arg in args]), args[0].batch_sizes)


# Model
class EncoderRNN(pl.LightningModule):

    def __init__(self, hidden_size, embed_size, embed):
        super(EncoderRNN, self).__init__()
        self.embeds = embed
        self.rnn = nn.GRU(embed_size, hidden_size, batch_first=True)

    def forward(self, inp):
        # emb = self.embed(inp)
        emb = elementwise_apply(self.embeds, inp)
        output, hidden = self.rnn(emb)
        return output, hidden


class DecoderRNN(pl.LightningModule):

    def __init__(self, embed, embed_size, hidden_size, output_size, max_len):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_len = max_len

        self.embeds = embed
        self.rnn = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward_setp(self, input, hidden, encoder_output):
        del encoder_output  # TODO(bzz): use it for Attention

        emb = elementwise_apply(self.embeds, input)
        # emb = self.embeds(input)

        o, h = self.rnn(emb, hidden)
        # o, _ = rnn.pad_packed_sequence(o,
        #                                batch_first=True,
        #                                padding_value=text_encoder.PAD_ID)
        # out = self.out(o)
        # return F.log_softmax(out, dim=-1), h
        out = elementwise_apply(self.out, o)
        return elementwise_apply(F.log_softmax, out, -1), h

    def forward(self, enc_h, enc_out, shifted_packed_tgt=None):
        """Uses teacher forcing, relies on tgt starting \w BOS"""
        # TODO(bzz): separate
        #  .train() that does require packed tgt
        #  .forward() that doesn't depend on tgt (and it's beeing packed)
        #  .infer() that does multiple .forward()/beams
        decoder_input = shifted_packed_tgt
        if shifted_packed_tgt is None:  # inference
            batch_size = enc_h.size(0) if enc_h is not None else 1
            decoder_input = torch.LongTensor(
                [batch_size * [text_encoder.BOS_ID]]).view(batch_size,
                                                           1).to(enc_h.device)
        decoder_hidden = enc_h  # TODO(bzz): transform/concat in case of N layers/2 directions

        decoder_output, decoder_hidden = self.forward_setp(
            decoder_input, decoder_hidden, enc_out)

        return decoder_output, decoder_hidden


class Seq2seqLightningModule(pl.LightningModule):

    def __init__(self, hp: Dict, enc: text_encoder.TextEncoder = None):
        super(Seq2seqLightningModule, self).__init__()
        self.enc = enc
        self.hparams = hp

        # share embedding layer by encoder and decoder
        self.embed = nn.Embedding(hp.vocab_size,
                                  hp.embedding_size,
                                  padding_idx=text_encoder.PAD_ID)
        # or pass in vocab_size and create encoder embeddings inside
        self.encoder = EncoderRNN(hp.hidden_size, hp.embedding_size, self.embed)
        self.decoder = DecoderRNN(self.embed, hp.embedding_size, hp.hidden_size,
                                  hp.vocab_size, hp.max_len)
        self.criterion = nn.NLLLoss(ignore_index=text_encoder.PAD_ID)

    def forward(self, src, tgt=None):
        encoder_output, encoder_hidden = self.encoder(src)
        outputs, hidden = self.decoder(encoder_hidden, encoder_output, tgt)
        return outputs

    def training_step(self, batch, batch_idx):  # REQUIRED
        src, tgt, shifted_tgt = batch
        # print(f"batch:{batch_idx}, y:{tgt.size()}, x:{src.size()}")
        # src_dec = self.enc.decode(src[0])[:40].replace('\n', '\\n')
        # print(f"\t {self.enc.decode(tgt[0])}, {src_dec}")

        output = self.forward(src, shifted_tgt)
        loss = self.criterion(output.data, tgt.data)  # both are packed

        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):  # OPTIONAL
        src, tgt, shifted_tgt = batch
        # print(f"batch:{batch_idx}, y:{tgt.size()}, x:{src.size()}")

        output = self.forward(src, shifted_tgt)
        loss = self.criterion(output.data, tgt.data)  # both are packed
        # loss = self.criterion(output.view(-1, output.shape[-1]), tgt.view(-1))

        # metrics
        preds = torch.argmax(output.data, dim=-1)
        # preds = elementwise_apply(torch.argmax, output, -1)
        (acc, tp, fp, fn) = metrics.acc_cm(preds, tgt.data, self.enc.vocab_size)

        preds_pad, _ = rnn.pad_packed_sequence(
            rnn.PackedSequence(preds, output.batch_sizes),
            batch_first=True,
            padding_value=text_encoder.PAD_ID)
        tgts_pad, _ = rnn.pad_packed_sequence(tgt,
                                              batch_first=True,
                                              padding_value=text_encoder.PAD_ID)

        bleu = metrics.compute_bleu(tgts_pad.tolist(), preds_pad.tolist())
        return {
            'val_loss': loss,
            'val_acc': acc,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'bleu': bleu
        }

    def validation_end(self, outputs):  # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tb_logs = {'val_loss': avg_loss, 'ppl': math.exp(avg_loss)}

        tb_logs['acc'] = torch.stack([x['val_acc'] for x in outputs]).mean()
        tb_logs['bleu'] = np.mean([x['bleu'] for x in outputs])

        total = {}
        for metric_name in ['tp', 'fp', 'fn']:
            metric_value = torch.stack([x[metric_name] for x in outputs]).sum()
            total[metric_name] = metric_value

        prec_rec_f1 = metrics.f1_score(total['tp'], total['fp'], total['fn'])
        tb_logs.update(prec_rec_f1)
        return {'avg_val_loss': avg_loss, 'log': tb_logs}

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_end(self, outputs):
        return self.validation_end(outputs)

    def configure_optimizers(self):  # REQUIRED
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        # return [optimizer], [scheduler]

    @pl.data_loader
    def train_dataloader(self):
        split = "test"  # TODO(bzz): change back to "train"
        ds = CodeSearchNetRAM(self.hparams.data_dir, self.enc, split)
        dl = DataLoader(
            ds,
            shuffle=True,  # False, if overfit_pct
            batch_size=self.hparams.batch_size,
            collate_fn=pack_collate)
        print(
            f"dataset:'{split}', size:{len(ds)}, batch:{self.hparams.batch_size}, nb_batches:{len(dl)}"
        )
        return dl

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(CodeSearchNetRAM(self.hparams.data_dir, self.enc,
                                           "valid"),
                          shuffle=False,
                          batch_size=self.hparams.batch_size,
                          collate_fn=pack_collate)

    @pl.data_loader
    def test_dataloader(self):
        return DataLoader(CodeSearchNetRAM(self.hparams.data_dir, self.enc,
                                           "test"),
                          shuffle=True,
                          batch_size=self.hparams.batch_size,
                          collate_fn=pack_collate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        """Specify the hyperparams for this LightningModule"""
        # MODEL specific
        parser = ArgumentParser(parents=[parent_parser],
                                formatter_class=ArgumentDefaultsHelpFormatter)
        parser.add_argument('--lr',
                            default=0.005,
                            type=float,
                            help="learning rate")
        parser.add_argument('--batch_size',
                            default=64,
                            type=int,
                            help="Size of the batch")
        parser.add_argument('--hidden_size',
                            default=128,
                            type=int,
                            help="Size of the hidden layers")
        parser.add_argument('--embedding_size',
                            default=100,
                            type=int,
                            help="Size of the (shared) embeddings")
        parser.add_argument('--max_len',
                            default=CodeSearchNetRAM.cut,
                            type=int,
                            help="Max sequence length")

        # training specific (for this model)
        parser.add_argument('--epochs',
                            default=30,
                            type=int,
                            help="Max number of expocs")
        return parser


def main(hparams):
    if hparams.inspect_data:
        print("Only show the input data")
        split_name = "test"
        enc = SubwordTextEncoder(vocab_filepath)
        ds = CodeSearchNetRAM(hparams.data_dir, enc, split_name)
        dl = DataLoader(ds,
                        batch_size=hparams.batch_size,
                        shuffle=True,
                        collate_fn=pack_collate)
        print(
            f"dataset:'{split_name}', size:{len(ds)}, batch:{hparams.batch_size}, nb_batches:{len(dl)}"
        )
        for i, batch in enumerate(dl, 1):
            fn_code, fn_name, fn_name_shft = (batch)
            print(
                f"\ny: {fn_name_shft.batch_sizes}, {fn_name.data.size()}, x:{fn_code.data.size()}"
            )
            print(
                f"y_pack: {fn_name_shft.batch_sizes}, {fn_name_shft.data.size()}"
            )

            # code_text = enc.decode(fn_code[0])[:60].replace("\n", "\\n")
            # print(f"\ty: -'{enc.decode(fn_name[0]):60}', x: '{code_text}'")

            code, c_lengths = rnn.pad_packed_sequence(
                fn_code, batch_first=True, padding_value=text_encoder.PAD_ID)
            name, n_lengths = rnn.pad_packed_sequence(
                fn_name, batch_first=True, padding_value=text_encoder.PAD_ID)
            name_shft, s_lengths = rnn.pad_packed_sequence(
                fn_name_shft,
                batch_first=True,
                padding_value=text_encoder.PAD_ID)

            print(
                f"name:{name.size()}, name_shift:{name_shft.size()}, code:{code.size()}"
            )

            for t in range(code.size(0)):
                code_text = enc.decode(code[t])[:60].replace("\n", "\\n")
                print(
                    f"\t-y: '{enc.decode(name[t][:60]):60}' x: '{code_text:60}', "
                )

                inp_text = enc.decode(name_shft[t])[:60].replace("\n", "\\n")
                print(f"\t+y: '{inp_text[:60]:60}'")

            if i == 1:
                break
        return

    if hparams.infer:
        print("Only running the inference")
        pretrained_model = Seq2seqLightningModule.load_from_checkpoint(
            checkpoint_path=hparams.model)
        pretrained_model.eval()
        pretrained_model.freeze()
        enc = SubwordTextEncoder(vocab_filepath)

        inp = "if (cp < replacementsLength) {\n      char[] chars = replacements[cp];"
        inp_enc = enc.encode(inp) + [text_encoder.EOS_ID]
        inp_vec = torch.LongTensor(inp_enc).unsqueeze_(0)
        # print(f"in:'{inp}'")
        # print(f"inp_enc:{inp_enc}")

        # predict
        output = pretrained_model(inp_vec)
        output = output.detach().squeeze(0)

        def generate(output: torch.Tensor, temperature=0.5) -> List[int]:
            res = []
            for i in range(output.size(0)):
                output_dist = output.data[i].div(temperature).exp()
                top_i = torch.multinomial(output_dist, 1)[0]
                if top_i == text_encoder.EOS_ID:
                    break
                res.append(top_i)
            return res

        out_enc = generate(output, 0.5)
        print(f"out_enc:{out_enc}")
        print(f"out:'{enc.decode(out_enc)}'")
        return

    if hparams.test:
        test_pct = 0.01
        print(f"Only runing evaluation on {test_pct*100}% of the test set")
        model = Seq2seqLightningModule.load_from_checkpoint(
            checkpoint_path=hparams.model)
        model.enc = SubwordTextEncoder(vocab_filepath)
        model.eval()

        trainer = pl.Trainer(test_percent_check=test_pct, gpus=hparams.gpus)
        trainer.test(model)
        return

    # Training
    enc = SubwordTextEncoder(vocab_filepath)
    hparams.vocab_size = enc.vocab_size
    model = Seq2seqLightningModule(hparams, enc)

    # TODO(bzz): load pre-trained embeddings (+ exclude from optimizer)
    # pretrained_embeddings = "<some vectors>"
    # model.embedding.weight.data.copy_(pretrained_embeddings)
    # model.embedding.weight.data[text_encoder.UNK_ID] = torch.zeros(embedding_dim)
    # model.embedding.weight.data[text_encoder.PAD_ID] = torch.zeros(embedding_dim)
    # model.embedding.weight.requires_grad = False

    trainer = pl.Trainer(max_nb_epochs=hparams.epochs,
                         fast_dev_run=False,
                         gpus=hparams.gpus)
                        #  early_stop_callback=None,
                        #  overfit_pct=0.00004)  # 1 example,1 batch_size, on test
    #  + batch_size=1, shuffle=False for proper overfitting
    trainer.fit(model)


if __name__ == "__main__":
    # project-wide training arguments, eg
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--data_dir',
                        type=str,
                        default="./data/codesearchnet/java",
                        help="Path to the unzip input data (CodeSearchNet)")

    parser.add_argument("--infer",
                        action="store_true",
                        help="Run the inference only")

    parser.add_argument("--test",
                        action="store_true",
                        help="Run the evaluation on the test set only")

    parser.add_argument("--model",
                        default=None,
                        type=str,
                        metavar='MODE_FILE',
                        help="Path to the serialized model .ckpt")

    parser.add_argument("--inspect_data",
                        action="store_true",
                        help="Inspect the DataLoader only")

    parser.add_argument('--gpus', type=int, default=0, help='How many gpus')

    # each LightningModule also defines arguments relevant to it
    parser = Seq2seqLightningModule.add_model_specific_args(parser)
    hyperparams = parser.parse_args()

    main(hyperparams)
