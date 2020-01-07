#!/usr/bin/env python3
# coding=utf-8
"""PyTorch implementation of char-level RNN for text generation in multiple lang
"""
from __future__ import unicode_literals, print_function, division

from typing import Dict, List, Tuple, Counter
from collections import Counter
from io import open

import glob
import os
import unicodedata
import string
import time
import random
import argparse
import logging

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO)
logger = logging.getLogger(__name__)

# build vocab
vocab: Counter = Counter()


# Read the data
def readLines(filename):
    content = open(filename, encoding='utf-8').read().strip()
    vocab.update(content)
    lines = content.split('\n')
    # for Eng, one could do [unicodeToAscii(line) for line in lines]
    return lines


category_lines: Dict[str, List[str]] = {}
all_categories: List[str] = []

for filename in glob.glob('data/*.txt'):
    category_tensor = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category_tensor)
    lines = readLines(filename)
    category_lines[category_tensor] = lines

n_categories = len(all_categories)
if n_categories == 0:
    raise RuntimeError('Data not found.')

print('# categories: {}, {} samples:'.format(n_categories, all_categories))
for c in all_categories:
    print("{}: {}".format(c, len(category_lines[c])))

# Reserved tokens for things like padding and EOS symbols.
PAD = "<pad>"
EOS = "<EOS>"
BOS = "<BOS>"
RESERVED_TOKENS = [PAD, EOS, BOS]
NUM_RESERVED_TOKENS = len(RESERVED_TOKENS)
PAD_ID = RESERVED_TOKENS.index(PAD)  # Normally 0
EOS_ID = RESERVED_TOKENS.index(EOS)  # Normally 1
BOS_ID = RESERVED_TOKENS.index(BOS)  # Normally 2

all_letters = " " * NUM_RESERVED_TOKENS + "".join(list(vocab))
n_letters = len(all_letters) + NUM_RESERVED_TOKENS

print("Total chars: {}".format(n_letters))
logger.debug("Vocab: {}".format(vocab))
logger.debug("Letters: '{}'".format(all_letters))


# One-hot vector for category
def encode_category(category):
    li = all_categories.index(category)
    tensor = torch.zeros(n_categories)
    tensor[li] = 1
    return tensor


# One-hot matrix of first to last letters (not including EOS) for input
def encode_chars(text: str) -> torch.Tensor:
    tensor = torch.zeros(len(text), n_letters)
    for ci in range(len(text)):
        char = text[ci]
        tensor[ci][all_letters.find(char)] = 1
    return tensor


# LongTensor of second letter to end (EOS) for target
def encode_shift_target(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(EOS_ID)  # n_letters - 1
    return torch.LongTensor(letter_indexes)


def decode(ids: torch.Tensor) -> str:
    res = []
    for one_hot in ids.tolist():
        # print(one_hot)
        for i in range(len(one_hot)):
            if one_hot[i] == 1:
                res.append(all_letters[i])
    return "".join(res)


t = "test"
logger.debug("in : '{}' - {}:".format(
    t,
    encode_chars(t).size()))  #\n{} encode_chars(t)
logger.debug("out: '{}'".format(decode(encode_chars(t))))
logger.debug("tgt: '{}'".format(encode_shift_target(t).size()))


class CityNames(Dataset):
    # cat_samples:List[Tuple[str, List[int]]] doesn't work https://github.com/python/mypy/issues/3955
    def __init__(self, cat_samples):
        input_targets = []
        for cat, samples in cat_samples:
            for inp in samples:
                input_targets.append((encode_category(cat), encode_chars(inp),
                                      encode_shift_target(inp)))
        self.data = input_targets

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


from torch.utils.data.dataloader import default_collate


def pad_collate(batch):
    (cats, inps, tgts) = zip(*batch)
    inps_pad = torch.nn.utils.rnn.pad_sequence(inps,
                                               batch_first=True,
                                               padding_value=PAD_ID)
    tgts_pad = torch.nn.utils.rnn.pad_sequence(tgts,
                                               batch_first=True,
                                               padding_value=PAD_ID)
    return default_collate(list(zip(*(cats, inps_pad, tgts_pad))))


train_set = CityNames(category_lines.items())
training_data_loader = DataLoader(train_set,
                                  shuffle=True,
                                  batch_size=64,
                                  collate_fn=pad_collate)

# for indexes in loader._index_sampler:
#     print([dataset[i] for i in indexes])
# import pdb; pdb.set_trace()

for iteration, s in enumerate(training_data_loader, 1):
    cat, inp, tgt = s[0], s[1], s[2]
    logger.debug("\t{}: (len({})=={}) cat:{}, int:{}, tgt:{}, ".format(
        iteration, type(s), len(s), cat.size(), inp.size(), tgt.size()))
    if iteration == 2:
        break
# import sys; sys.exit()

## Model

MODEL_SNAPSHOT = 'conditional-char-rnn-ds.pt'


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
            # print("category:{} input:{} i:{} hidden:{}".format(
            #     category.size(), inp.size(), inp[:, i].size(), h.size()))

            input_combined = torch.cat((category, inp[:, i], h), 1)
            h = self.i2h(input_combined)
            output = self.i2o(input_combined)
            output_combined = torch.cat((h, output), 1)
            res.append(self.o2o(output_combined))

        return torch.stack(res, dim=1).view(-1, self.output_size)
        # return self.o2o(output_combined)


## Train
def train():
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")
    print("using '{}'".format(device))

    n_epochs = 1000
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

            for batch_ndx, s in enumerate(training_data_loader, 1):
                category_tensor, input_tensor, target_tensor = s[0].to(
                    device), s[1].to(device), s[2].to(device)

                loss: torch.Tensor = 0

                output = rnn(category_tensor, input_tensor)
                # print(" category:{} input:{} target:{} output:{}".format(
                #     category_tensor.size(), input_tensor.size(),
                #     target_tensor.size(), output.size()))

                loss = criterion(output, target_tensor.view(-1))

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                loss_avg += loss.data / input_tensor.size()[0]

                if batch_ndx % print_every == 0:
                    print('{:.0f}s ({} {:.0f}%) loss: {:.4f}'.format(
                        time.time() - start, epoch,
                        batch_ndx / len(training_data_loader) * 100, loss))

                if batch_ndx % plot_every == 0:
                    all_losses.append(loss_avg / plot_every)
                    loss_avg = 0

    except KeyboardInterrupt:
        pass
    finally:
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
        category_input = encode_category(category).unsqueeze(0)
        chars_input = encode_chars(start_char).unsqueeze(0)

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
            if top_i == EOS_ID:
                break
            else:
                char = all_letters[top_i]
                output_str += char
                chars_input = encode_chars(output_str).unsqueeze(0)

        return output_str

    def generate(category, start_chars='ABC'):
        for start_char in start_chars:
            print(generate_one(category, start_char))

    generate('us', 'ENG')
    generate('ru', 'РУС')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",
                        action='store_true',
                        help="Run the training")
    parser.add_argument("-v",
                        dest="debug",
                        action='store_true',
                        help="Verbose logging")
    args = parser.parse_args()
    # TODO(bzz): set seed

    if args.debug:
        logger.setLevel(logging.DEBUG)

    if args.train:
        train()

    inference()


if __name__ == "__main__":
    main()