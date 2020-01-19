#!/usr/bin/env python3
# coding=utf-8
"""
PyTorch-lightning (PTL) implementation of seq2seq architecture for 
function name suggestions.
"""
from typing import Dict, List, Tuple

import os
import glob
import time
import logging
from argparse import ArgumentParser
from pathlib import Path

from data_generators import text_encoder
from data_generators.text_encoder import TextEncoder, SubwordTextEncoder

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import pytorch_lightning as pl

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
        return (torch.LongTensor(self.enc.encode(row["code"][:self.cut])),
                torch.LongTensor(self.enc.encode(row["func_name"][:self.cut])))

    def __len__(self):
        return len(self.pd)


def pad_collate(batch):
    """Padds input and target to the same length"""
    (fn_code, fn_name) = zip(*batch)

    inps_pad = torch.nn.utils.rnn.pad_sequence(
        fn_code, batch_first=True, padding_value=text_encoder.PAD_ID)
    tgts_pad = torch.nn.utils.rnn.pad_sequence(
        fn_name, batch_first=True, padding_value=text_encoder.PAD_ID)

    return torch.utils.data.dataloader.default_collate(
        list(zip(*(inps_pad, tgts_pad))))


# Model
class EncoderRNN(pl.LightningModule):

    def __init__(self, hidden_size, embed_size, embed):
        super(EncoderRNN, self).__init__()
        self.embed = embed
        self.rnn = nn.GRU(embed_size, hidden_size, batch_first=True)

    def forward(self, inp):
        emb = self.embed(inp)
        output, hidden = self.rnn(emb)
        return output, hidden


class DecoderRNN(pl.LightningModule):

    def __init__(self, embed, embed_size, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.embed = embed
        self.rnn = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward_setp(self, input, hidden, encoder_output):
        del encoder_output  # TODO(bzz): use it for Attention

        emb = self.embed(input)
        # print(f"\temb:{emb.size()}, hid:{hidden.size()}")
        o, h = self.rnn(emb, hidden)
        out = self.out(o)
        # print(f"\t\trnn_out:{o.size()}, rnn_hid:{h.size()}, out:{out.size()}")
        return F.log_softmax(out, dim=1), h

    def forward(self, inp, enc_h, enc_out):
        batch_size = inp.size(0)
        max_seq_len = inp.size(1)

        decoder_outputs = []
        decoder_hidden = enc_h  # TODO(bzz): transform/concat in case of N layers/2 directions

        # TODO(bzz): add teacher-forcing like e.g
        # https://github.com/IBM/pytorch-seq2seq/blob/f146087a9a271e9b50f46561e090324764b081fb/seq2seq/models/DecoderRNN.py#L140
        decoder_input = torch.LongTensor([batch_size * [text_encoder.BOS_ID]
                                         ]).view(batch_size, 1).to(inp.device)
        # print(f"dec_inp:{decoder_input.size()}, dec_h:{decoder_hidden.size()}, enc_out:{enc_out.size()}")

        for t in range(max_seq_len):  # TODO(bzz): unroll in graph&compare
            decoder_output, decoder_hidden = self.forward_setp(
                decoder_input, decoder_hidden, enc_out)

            step_output = decoder_output.squeeze(1)
            decoder_outputs.append(step_output)
            decoder_input = self._decode(t, step_output)

            # print(f"\t\tdecoded output:{len(decoder_outputs)}, next:{decoder_input.size()}")

        return torch.stack(decoder_outputs, dim=1), decoder_hidden

    @staticmethod
    def _decode(step: int, step_output):
        ids = step_output.topk(1)[1]
        return ids


class Seq2seqLightningModule(pl.LightningModule):

    def __init__(self, enc: text_encoder.TextEncoder, hp: Dict):
        super(Seq2seqLightningModule, self).__init__()
        self.enc = enc
        self.hparams = hp

        # share embedding layer by encoder and decoder
        self.embed = nn.Embedding(enc.vocab_size, hp.embedding_size)
        # or pass in vocab_size and create encoder embeddings inside
        self.encoder = EncoderRNN(hp.hidden_size, hp.embedding_size, self.embed)
        self.decoder = DecoderRNN(self.embed, hp.embedding_size, hp.hidden_size,
                                  enc.vocab_size)
        self.criterion = nn.NLLLoss()

    def forward(self, src, tgt):
        encoder_output, encoder_hidden = self.encoder(src)
        # print(f"src:{src.size()}, tgt:{tgt.size()}, enc_out:{encoder_output.size()}, enc_h:{encoder_hidden.size()}")
        outputs, hidden = self.decoder(tgt, encoder_hidden, encoder_output)
        return outputs

    def training_step(self, batch, batch_idx):  # REQUIRED
        src, tgt = batch
        print(
            f"{batch_idx} - y:{self.enc.decode(tgt[0])} {tgt.size()}, x:{src.size()}"
        )
        output = self.forward(src, tgt)

        output = output.view(-1, output.shape[-1])
        target = tgt.view(-1)
        loss = self.criterion(output, target)

        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):  # OPTIONAL
        src, tgt = batch
        output = self.forward(src, tgt)
        return {
            'val_loss':
                self.criterion(output.view(-1, output.shape[-1]), tgt.view(-1))
        }

    def validation_end(self, outputs):  # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):  # REQUIRED
        return torch.optim.Adam(self.parameters(),
                                lr=self.hparams.learning_rate)

    @pl.data_loader
    def train_dataloader(self):
        # TODO(bzz): change back to slower train
        return DataLoader(
            CodeSearchNetRAM(self.hparams.data_dir, self.enc, "test"),
            shuffle=False,  # False, if no overfit_pct
            batch_size=self.hparams.batch_size,
            collate_fn=pad_collate)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(CodeSearchNetRAM(self.hparams.data_dir, self.enc,
                                           "valid"),
                          shuffle=False,
                          batch_size=2,
                          collate_fn=pad_collate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        """Specify the hyperparams for this LightningModule"""
        # MODEL specific
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('--learning_rate', default=0.005, type=float)
        parser.add_argument('--batch_size', default=1, type=int)
        parser.add_argument('--hidden_size', default=128, type=int)
        parser.add_argument('--embedding_size', default=100, type=int)

        # training specific (for this model)
        parser.add_argument('--epochs', default=10, type=int)
        return parser


def main(hparams):
    if hparams.inspect_data:
        print("Only testing the data preprocessing")
        split_name = "test"
        enc = SubwordTextEncoder(vocab_filepath)
        ds = CodeSearchNetRAM(hparams.data_dir, enc, split_name)
        dl = DataLoader(ds,
                        batch_size=hparams.batch_size,
                        shuffle=True,
                        collate_fn=pad_collate)
        print(
            f"dataset:'{split_name}', size:{len(ds)}, batch:{hparams.batch_size}, nb_batches:{len(dl)}"
        )
        for i, batch in enumerate(dl, 1):
            fn_code, fn_name = (batch)
            print("{} - y:{} {}, x:{}".format(i, enc.decode(fn_name[0]),
                                              fn_name.size(),
                                              fn_code.size()))  # str()[:10]
            if i == 2:
                break
        return

    if hparams.infer:
        # TODO(bzz): load from checkpoint, run inference
        return

    # Training
    enc = SubwordTextEncoder(vocab_filepath)
    model = Seq2seqLightningModule(enc, hparams)
    # TODO(bzz): load pre-trained embeddings (+ exclude from optimizer)
    # pretrained_embeddings = "<some vectors>"
    # model.embedding.weight.data.copy_(pretrained_embeddings)
    # model.embedding.weight.data[text_encoder.UNK_ID] = torch.zeros(embedding_dim)
    # model.embedding.weight.data[text_encoder.PAD_ID] = torch.zeros(embedding_dim)
    # model.embedding.weight.requires_grad = False
    trainer = pl.Trainer(max_nb_epochs=hparams.epochs,
                         fast_dev_run=False,
                         early_stop_callback=None,
                         overfit_pct=0.00004)  # 1 example,1 batch_size, on test
    trainer.fit(model)


if __name__ == "__main__":
    # project-wide training arguments, eg
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--data_dir', type=str, default="./data/codesearchnet")
    # parser.add_argument('--gpus', type=str, default=None)

    parser.add_argument("--infer",
                        action="store_true",
                        help="Run the inference only")

    parser.add_argument("--inspect_data",
                        action="store_true",
                        help="Inspect the DataLoader only")

    # each LightningModule also defines arguments relevant to it
    parser = Seq2seqLightningModule.add_model_specific_args(parser)
    hyperparams = parser.parse_args()

    main(hyperparams)
