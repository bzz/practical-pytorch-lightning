# coding=utf-8
"""Helpers for data preprocessing
"""

import glob
import os
from collections import Counter
from typing import Dict, List, Tuple

import torch
from torch.utils.data.dataloader import default_collate

# Reserved tokens for things like padding and EOS symbols.
PAD = "<pad>"
EOS = "<EOS>"
BOS = "<BOS>"
RESERVED_TOKENS = [PAD, EOS, BOS]
NUM_RESERVED_TOKENS = len(RESERVED_TOKENS)
PAD_ID = RESERVED_TOKENS.index(PAD)  # Normally 0
EOS_ID = RESERVED_TOKENS.index(EOS)  # Normally 1
BOS_ID = RESERVED_TOKENS.index(BOS)  # Normally 2


class Encoder(object):
    def __init__(self, all_letters, all_categories):
        self.all_letters = all_letters  # public
        self.all_categories = all_categories
        self.n_letters = len(all_letters)
        self.n_categories = len(all_categories)

    # One-hot vector for category
    def encode_category(self, category) -> torch.Tensor:
        li = self.all_categories.index(category)
        tensor = torch.zeros(self.n_categories)
        tensor[li] = 1
        return tensor

    # One-hot matrix of first to last letters (not including EOS) for input
    def encode_chars(self, text: str) -> torch.Tensor:
        tensor = torch.zeros(len(text), self.n_letters)
        for ci in range(len(text)):
            char = text[ci]
            tensor[ci][self.all_letters.find(char)] = 1
        return tensor

    # LongTensor of second letter to end (EOS) for target
    def encode_shift_target(self, line) -> torch.LongTensor:
        letter_indexes = [
            self.all_letters.find(line[li]) for li in range(1, len(line))
        ]
        letter_indexes.append(EOS_ID)  # n_letters - 1
        return torch.LongTensor(letter_indexes)

    def decode(self, ids: torch.Tensor) -> str:
        res = []
        for one_hot in ids.tolist():
            # print(one_hot)
            for i in range(len(one_hot)):
                if one_hot[i] == 1:
                    res.append(self.all_letters[i])
        return "".join(res)


def load(from_path: str) -> Tuple[Encoder, List[Tuple[str, str]]]:
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

    for filename in glob.glob(from_path):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = readLines(filename)
        category_lines[category] = lines

    n_categories = len(all_categories)
    if n_categories == 0:
        raise RuntimeError('Data not found.')

    print('# categories: {}, {} samples:'.format(n_categories, all_categories))
    for c in all_categories:
        print("{}: {}".format(c, len(category_lines[c])))

    all_letters = " " * NUM_RESERVED_TOKENS + "".join(list(vocab))
    print("Total chars: {}".format(len(all_letters)))

    # TODO(bzz): inject logger dependency
    # logger.debug("Vocab: {}".format(vocab))
    # logger.debug("Letters: '{}'".format(all_letters))

    encoder = Encoder(all_letters, all_categories)
    total_samples = [(k, v) for k, vs in category_lines.items() for v in vs]
    return encoder, total_samples


def pad_collate(batch):
    """Padds input and target to the same length"""
    (cats, inps, tgts) = zip(*batch)
    inps_pad = torch.nn.utils.rnn.pad_sequence(inps,
                                               batch_first=True,
                                               padding_value=PAD_ID)
    tgts_pad = torch.nn.utils.rnn.pad_sequence(tgts,
                                               batch_first=True,
                                               padding_value=PAD_ID)
    return default_collate(list(zip(*(cats, inps_pad, tgts_pad))))
