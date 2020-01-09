# coding=utf-8
"""Helpers for data preprocessing
"""

import glob
import os
from collections import Counter
from typing import Dict, List, Tuple

import torch
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset

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
    """All vocabularies used for encoding input, category and target"""
    def __init__(self, all_letters: str, all_categories: List[str]):
        self.all_letters = all_letters  # public
        self.all_categories = all_categories
        self.n_letters = len(all_letters)
        self.n_categories = len(all_categories)

    # One-hot vector for category
    def one_hot_category(self, category) -> torch.Tensor:
        li = self.all_categories.index(category)
        tensor = torch.zeros(self.n_categories)
        tensor[li] = 1
        return tensor

    # One-hot matrix of first to last letters (not including EOS) for input
    def one_hot_chars(self, text: str) -> torch.Tensor:
        tensor = torch.zeros(len(text), self.n_letters)
        for ci in range(len(text)):
            char = text[ci]
            tensor[ci][self.all_letters.find(char)] = 1
        return tensor

    def decode_one_hot(self, ids: torch.Tensor) -> str:
        res = []
        for one_hot in ids.tolist():
            for i in range(len(one_hot)):
                if one_hot[i] == 1:
                    res.append(self.all_letters[i])
        return "".join(res)

    # LongTensor of second letter to end + EOS for the target
    def encode_shift_target(self, line) -> torch.LongTensor:
        ids = [self.all_letters.find(line[li]) for li in range(1, len(line))]
        ids.append(EOS_ID)
        return torch.LongTensor(ids)

    def encode_category(self, category: str) -> torch.LongTensor:
        id = [self.all_categories.index(category)]
        return torch.LongTensor(id)

    def encode(self, line) -> torch.LongTensor:
        ids = [self.all_letters.find(c) for c in line]
        return torch.LongTensor(ids)

    def decode(self, ids) -> str:
        return "".join(map(lambda id: self.all_letters[id], ids))


class CityNames(Dataset):
    """Base class for converting from text strings to vectors."""
    def to_tensor(
        self, s: Tuple[str, str]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Transform a human-readable strings for category and text into vectors."""
        raise NotImplementedError()

    #  .items() as cat_sampels does not work https://github.com/python/mypy/issues/3955
    def __init__(self, enc: Encoder, cat_sample: List[Tuple[str, str]]):
        self.enc = enc
        self.data = list(map(self.to_tensor, cat_sample))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class CityNamesOneHot(CityNames):
    """1-hot encoding for both, category and text."""
    def to_tensor(
        self, s: Tuple[str, str]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        (cat, inp) = s
        return (self.enc.one_hot_category(cat), self.enc.one_hot_chars(inp),
                self.enc.encode_shift_target(inp))


class CityNamesIDs(CityNames):
    """Sequence of int ids for"""
    def to_tensor(
        self, s: Tuple[str, str]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        (cat, inp) = s
        return (
            # category for each letter
            self.enc.encode_category(cat).expand(len(inp)),
            self.enc.encode(inp),
            self.enc.encode_shift_target(inp))


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
    if cats[0].size() == inps[0].size():
        cats_pad = torch.nn.utils.rnn.pad_sequence(cats,
                                                   batch_first=True,
                                                   padding_value=PAD_ID)
    else: # do not pad in 1-hot case
        cats_pad = cats

    inps_pad = torch.nn.utils.rnn.pad_sequence(inps,
                                               batch_first=True,
                                               padding_value=PAD_ID)
    tgts_pad = torch.nn.utils.rnn.pad_sequence(tgts,
                                               batch_first=True,
                                               padding_value=PAD_ID)
    return default_collate(list(zip(*(cats_pad, inps_pad, tgts_pad))))


def test_encoder():
    e = Encoder("tes one", ["ru", "eng"])
    t = "test one"
    c = "ru"
    print("Testing encoder: decode(encode('{}'))".format(t))
    print("\tin : '{}' - {}:".format(t, e.encode(t).size()))
    print("\tout: '{}'".format(e.decode(e.encode(t))))
    print("\ttgt: '{}'".format(e.encode_shift_target(t).size()))
    print("\tcat: '{}' {}".format(c, e.one_hot_category(c)))


def test_one_hot():
    e = Encoder("tes one", ["ru", "eng"])
    t = "test one"
    c = "eng"
    print("Testing encoder: decode_one_hot(one_hot_chars('{}'))".format(t))
    print("\tin : '{}' - {}:".format(t, e.one_hot_chars(t).size()))
    print("\tout: '{}'".format(e.decode_one_hot(e.one_hot_chars(t))))
    print("\tcat: '{}' {}".format(c, e.encode_category(c)))


if __name__ == "__main__":
    # test encoder/decoder
    test_encoder()
    test_one_hot()


# TODO(bzz): move to a better place
def generate_one(net,
                 encoder,
                 category,
                 start_char='A',
                 temperature=0.5,
                 max_length=20):
    category_input = encoder.encode_category(category).unsqueeze(0)
    chars_input = encoder.encode_chars(start_char).unsqueeze(0)

    output_str = start_char
    # logger.debug("start inferense: '%s' = '%s'", start_char,
    #              chars_input.size())

    for i in range(max_length):
        output = net(category_input, chars_input)
        # logger.debug("next prediction: '%s'", output.size())

        # Sample as a multinomial distribution
        output_dist = output.data[-1].div(temperature).exp()
        # logger.debug("next prediction: '%s'", output_dist.size())

        top_i = torch.multinomial(output_dist, 1)[0]
        # logger.debug("next prediction argmax: '%d'", top_i)

        # Stop at EOS, or add to output_str
        if top_i == EOS_ID:
            break
        else:
            char = encoder.all_letters[top_i]
            output_str += char
            chars_input = encoder.encode_chars(output_str).unsqueeze(0)

    return output_str


def generate(net, encoder, category, start_chars='ABC', max_length=20):
    for start_char in start_chars:
        print(generate_one(net, encoder, category, start_char, max_length))
