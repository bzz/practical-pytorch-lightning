#!/usr/bin/env python3
# coding=utf-8
"""PyTorch implementation of char-level RNN for text generation in multiple lang
"""
from __future__ import unicode_literals, print_function, division

from typing import Dict, List, Tuple, Counter
from io import open

import glob
import os
import unicodedata
import string
import time
import random
import math
import argparse
import logging

import data

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

SEED = 2334
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO)
logger = logging.getLogger(__name__)

# TODO(bzz): cache vocabulary, load full data only in train()
encoder, total_samples = data.load('data/*.txt')

n_letters = encoder.n_letters  # used in the model
n_categories = encoder.n_categories

# test: encode/decode
t = "test"
logger.debug("in : '{}' - {}:".format(
    t,
    encoder.one_hot_chars(t).size()))  #\n{} one_hot_chars(t)
logger.debug("out: '{}'".format(encoder.decode_one_hot(encoder.one_hot_chars(t))))
logger.debug("tgt: '{}'".format(encoder.encode_shift_target(t).size()))

## Model

MODEL_SNAPSHOT = 'conditional-char-rnn.pt'


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.i2h = nn.Linear(n_categories + input_size + hidden_size,
                             hidden_size)
        self.i2o = nn.Linear(n_categories + input_size + hidden_size,
                             output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)

    def forward(self, category, inp):
        h = torch.zeros(inp.shape[0], self.hidden_size).to(device=inp.device)

        res = []
        for i in range(inp.size()[1]):
            input_combined = torch.cat((category, inp[:, i], h), 1)
            h = self.i2h(input_combined)
            output = self.i2o(input_combined)
            output_combined = torch.cat((h, output), 1)
            res.append(self.o2o(output_combined))

        return torch.stack(res, dim=1).view(-1, self.output_size)


## Train
def train(n_epochs: int):
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")
    print("using '{}'".format(device))

    global total_samples
    train_samples, val_samples = np.split(total_samples,
                                          [int(.9 * len(total_samples))])

    logger.info("Total samples:%d = train:%d, valid:%d", len(total_samples),
                len(train_samples), len(val_samples))
    del total_samples

    train_set = data.CityNamesOneHot(encoder, train_samples)
    training_data_loader = DataLoader(train_set,
                                      shuffle=True,
                                      batch_size=64,
                                      collate_fn=data.pad_collate)

    val_set = data.CityNamesOneHot(encoder, val_samples)
    val_data_loader = DataLoader(val_set,
                                 shuffle=False,
                                 batch_size=2,
                                 collate_fn=data.pad_collate)

    # debug output
    for iteration, s in enumerate(training_data_loader, 1):
        cat, inp, tgt = s[0], s[1], s[2]
        logger.debug("\t{}: (len({})=={}) cat:{}, int:{}, tgt:{}, ".format(
            iteration, type(s), len(s), cat.size(), inp.size(), tgt.size()))
        if iteration == 2:
            break

    print_every = 20
    plot_every = 10
    all_losses = []
    loss_avg = 0  # Zero every plot_every epochs to keep a running average
    learning_rate = 0.0005

    rnn = RNN(n_letters, 128, n_letters).to(device)

    optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    try:
        start = time.time()
        print("Training for {} epochs...".format(n_epochs))
        for epoch in range(1, n_epochs + 1):

            # train
            for batch_ndx, s in enumerate(training_data_loader, 1):
                category_tensor, input_tensor, target_tensor = s[0].to(
                    device), s[1].to(device), s[2].to(device)

                output = rnn(category_tensor, input_tensor)
                loss = criterion(output, target_tensor.view(-1))

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                loss_avg += loss.data / input_tensor.size()[0]

                if batch_ndx % print_every == 0:
                    print('{:.0f}s ({} {:.0f}%) loss: {:.4f}, ppl: {:8.2f}'.
                          format(time.time() - start, epoch,
                                 batch_ndx / len(training_data_loader) * 100,
                                 loss_avg, math.exp(loss_avg)))

                if batch_ndx % plot_every == 0:
                    all_losses.append(loss_avg / plot_every)
                    loss_avg = 0

            # test
            with torch.no_grad():
                rnn.eval()
                test_loss_avg = 0
                for batch in val_data_loader:
                    cat, inp, tgt = batch[0].to(device), batch[1].to(
                        device), batch[2].to(device)

                    out = rnn(cat, inp)
                    test_loss = criterion(out, tgt.view(-1))
                    test_loss_avg += test_loss.data / inp.size()[0]

            print("\t test ppl: {:.4f} ".format(
                math.exp(test_loss_avg / len(val_data_loader))))

    except KeyboardInterrupt:
        pass
    finally:
        # checkpoint
        print("Saving the model to '{}'".format(MODEL_SNAPSHOT))
        torch.save(rnn.state_dict(), MODEL_SNAPSHOT)

    # Plot
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    # %matplotlib inline

    plt.figure()
    plt.plot(all_losses)


# Sample
def inference():
    print("Loading the model from '{}'".format(MODEL_SNAPSHOT))
    net = RNN(n_letters, 128, n_letters)
    net.load_state_dict(torch.load(MODEL_SNAPSHOT))

    max_length = 20

    def generate_one(category, start_char='A', temperature=0.5):
        category_input = encoder.one_hot_category(category).unsqueeze(0)
        chars_input = encoder.one_hot_chars(start_char).unsqueeze(0)

        output_str = start_char
        logger.debug("start inferense: '%s' = '%s'", start_char,
                     chars_input.size())

        for i in range(max_length):
            output = net(category_input, chars_input)
            logger.debug("next prediction: '%s'", output.size())

            # Sample as a multinomial distribution
            output_dist = output.data[-1].div(temperature).exp()
            logger.debug("next prediction: '%s'", output_dist.size())

            top_i = torch.multinomial(output_dist, 1)[0]
            logger.debug("next prediction argmax: '%d'", top_i)

            # Stop at EOS, or add to output_str
            if top_i == data.EOS_ID:
                break
            else:
                char = encoder.all_letters[top_i]
                output_str += char
                chars_input = encoder.one_hot_chars(output_str).unsqueeze(0)

        return output_str

    def generate(category, start_chars='ABC'):
        for start_char in start_chars:
            print(generate_one(category, start_char))

    generate('us', 'ENG')
    generate('ru', 'РУС')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",
                        action="store_true",
                        help="Run the training")
    parser.add_argument("-v",
                        dest="debug",
                        action="store_true",
                        help="Verbose logging")
    parser.add_argument("--epoch",
                        type=int,
                        default=10,
                        help="Number of epoch to train")

    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    if args.train:
        train(args.epoch)

    inference()


if __name__ == "__main__":
    main()