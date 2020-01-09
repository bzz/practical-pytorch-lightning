#!/usr/bin/env python3
# coding=utf-8
"""PyTorch-lightning (PTL) implementation of char-level RNN for text generation.
"""
import os
from argparse import ArgumentParser

import data

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, IterableDataset

from pytorch_lightning import Trainer
import pytorch_lightning as pl

SEED = 2334
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

encoder, train_samples, val_samples = data.Encoder, [], []


class LightningRNN(pl.LightningModule):
    def __init__(self, hp):
        super(LightningRNN, self).__init__()
        self.hparams = hp

        self.i2h = torch.nn.Linear(
            hp.n_categories + hp.input_size + hp.hidden_size, hp.hidden_size)
        self.i2o = torch.nn.Linear(
            hp.n_categories + hp.input_size + hp.hidden_size, hp.output_size)
        self.o2o = torch.nn.Linear(hp.hidden_size + hp.output_size,
                                   hp.output_size)

    def forward(self, category, inp):
        h = torch.zeros(inp.shape[0],
                        self.hparams.hidden_size).to(device=inp.device)

        res = []
        for i in range(inp.size()[1]):
            input_combined = torch.cat((category, inp[:, i], h), 1)
            h = self.i2h(input_combined)
            output = self.i2o(input_combined)
            output_combined = torch.cat((h, output), 1)
            res.append(self.o2o(output_combined))

        return torch.stack(res, dim=1).view(-1, self.hparams.output_size)

    def training_step(self, batch, batch_nb):  # REQUIRED
        cat, x, y = batch
        y_hat = self.forward(cat, x)
        loss = F.cross_entropy(y_hat, y.view(-1))
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):  # OPTIONAL
        cat, x, y = batch
        y_hat = self.forward(cat, x)
        return {'val_loss': F.cross_entropy(y_hat, y.view(-1))}

    def validation_end(self, outputs):  # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):  # REQUIRED
        return torch.optim.Adam(self.parameters(),
                                lr=self.hparams.learning_rate)

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(data.CityNames(encoder, train_samples),
                          shuffle=True,
                          batch_size=self.hparams.batch_size,
                          collate_fn=data.pad_collate)
        # TODO(bzz): what logic belongs to the Dataset vs Problem definition?
        #  - download
        #  - train/val split?
        #  - toTensor() (using Eencoder)?
        #return DataLoader(CityNames(train=True), batch_size=32)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(data.CityNames(encoder, val_samples),
                          shuffle=False,
                          batch_size=2,
                          collate_fn=data.pad_collate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the hyperparams for this LightningModule
        """
        # MODEL specific
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('--learning_rate', default=0.005, type=float)
        parser.add_argument('--batch_size', default=64, type=int)
        parser.add_argument('--hidden_size', default=128, type=int)
        # TODO(bzz): save hparams set as a GIN templates, to check-in

        # training specific (for this model)
        parser.add_argument('--epochs', default=100, type=int)

        return parser


def main(hparams):
    global encoder, train_samples, val_samples

    encoder, total_samples = data.load('data/*.txt')
    train_samples, val_samples = np.split(total_samples,
                                          [int(.9 * len(total_samples))])
    print("Total samples:{} = train:{}, valid:{}".format(
        len(total_samples), len(train_samples), len(val_samples)))
    del total_samples

    hparams.n_categories = len(encoder.all_categories)
    hparams.input_size = len(encoder.all_letters)
    hparams.output_size = len(encoder.all_letters)

    # runs the main training/val loop, etc...
    model = LightningRNN(hparams)
    trainer = Trainer(max_nb_epochs=hparams.epochs, fast_dev_run=False)
    trainer.fit(model)


if __name__ == '__main__':
    # project-wide training arguments, eg
    # root_dir = os.path.dirname(os.path.realpath(__file__))
    parent_parser = ArgumentParser(add_help=False)

    # each LightningModule also defines arguments relevant to it
    parser = LightningRNN.add_model_specific_args(parent_parser)
    hyperparams = parser.parse_args()

    main(hyperparams)
