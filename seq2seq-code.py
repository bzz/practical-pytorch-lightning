#!/usr/bin/env python3
# coding=utf-8
"""
PyTorch-lightning (PTL) implementation of seq2seq architecture for
function name suggestions.
"""
import glob
import logging
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
from torch.utils.data import DataLoader, Dataset

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

    def __init__(self, embed, embed_size, hidden_size, output_size, max_len):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_len = max_len

        self.embed = embed
        self.rnn = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward_setp(self, input, hidden, encoder_output):
        del encoder_output  # TODO(bzz): use it for Attention

        emb = self.embed(input)
        o, h = self.rnn(emb, hidden)
        out = self.out(o)
        return F.log_softmax(out, dim=-1), h

    def forward(self, enc_h, enc_out, tgt=None):
        batch_size = tgt.size(0) if tgt is not None else 1
        max_seq_len = tgt.size(1) if tgt is not None else self.max_len

        decoder_outputs = []
        decoder_hidden = enc_h  # TODO(bzz): transform/concat in case of N layers/2 directions

        # TODO(bzz): add teacher-forcing like e.g
        # https://github.com/IBM/pytorch-seq2seq/blob/f146087a9a271e9b50f46561e090324764b081fb/seq2seq/models/DecoderRNN.py#L140
        decoder_input = torch.LongTensor([batch_size * [text_encoder.BOS_ID]
                                         ]).view(batch_size, 1).to(enc_h.device)
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

    def __init__(self, hp: Dict, enc: text_encoder.TextEncoder = None):
        super(Seq2seqLightningModule, self).__init__()
        self.enc = enc
        self.hparams = hp

        # share embedding layer by encoder and decoder
        self.embed = nn.Embedding(hp.vocab_size, hp.embedding_size)
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
        src, tgt = batch
        output = self.forward(src, tgt)

        # print(f"{batch_idx} - y:{tgt.size()}, x:{src.size()}")
        # src_dec = self.enc.decode(src[0])[:40].replace('\n', '\\n')
        # print(f"\t {self.enc.decode(tgt[0])}, {src_dec}")

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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
        #                                                        T_max=10)
        # return [optimizer], [scheduler]

    @pl.data_loader
    def train_dataloader(self):
        split = "test"  # TODO(bzz): change back to "train"
        ds = CodeSearchNetRAM(self.hparams.data_dir, self.enc, split)
        dl = DataLoader(
            ds,
            shuffle=True,  # False, if overfit_pct
            batch_size=self.hparams.batch_size,
            collate_fn=pad_collate)
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
                          collate_fn=pad_collate)

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
                            default=100,
                            type=int,
                            help="Max number of expocs")
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
            code = enc.decode(fn_code[0])[:30].replace("\n", "\\n")
            print(f"{i} y:{enc.decode(fn_name[0])} {fn_name.size()}, x:{code}")
            if i == 4:
                break
        return

    if hparams.infer:
        print("Only running the inference")
        pretrained_model = Seq2seqLightningModule.load_from_checkpoint(
            checkpoint_path=hparams.infer)
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
    # parser.add_argument('--gpus', type=str, default=None)

    parser.add_argument("--infer",
                        default=None,
                        type=str,
                        metavar='MODE_FILE',
                        help="Run the inference only")

    parser.add_argument("--inspect_data",
                        action="store_true",
                        help="Inspect the DataLoader only")

    parser.add_argument('--gpus', type=int, default=0, help='How many gpus')

    # each LightningModule also defines arguments relevant to it
    parser = Seq2seqLightningModule.add_model_specific_args(parser)
    hyperparams = parser.parse_args()

    main(hyperparams)
