# coding=utf-8
"""Builds PBE-like subword vocabulary
"""

import argparse

import pandas as pd
import tqdm

from data_generators import text_encoder


def read_fns_codesearchnet(path):
    # TODO(bzz): list files
    files = []
    for filepath in tqdm(sorted(files)):
        for chunk in pd.read_json(filepath,
                                  orient='records',
                                  compression='gzip',
                                  lines=True,
                                  chunksize=100):
            for i, row in chunk.iterrows():
                func_name = row["func_name"]
                code = row["code"]
                yield code


def main(args):
    encoder = text_encoder.SubwordTextEncoder()

    fns = read_fns_codesearchnet(args.data)
    encoder.build_from_generator(fns, args.min_count, args.num_iterations)
    encoder.store_to_file(args.output_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)

    # tf.flags.DEFINE_integer('corpus_max_lines', 10000,
    #                         'How many lines of corpus to read')

    parser.add_argument('--data',
                        default='./codesearchnet',
                        type=str,
                        help='where to read *.jsonl from')

    parser.add_argument('--output_filename',
                        default='/tmp/my.subword_text_encoder',
                        type=str,
                        help='where to store the SubwordTextEncoder')
    parser.add_argument('--min_count',
                        default=100,
                        type=int,
                        help='Minimum subtoken count in corpus')
    parser.add_argument('--num_iterations',
                        default=100,
                        type=int,
                        help='Number of iterations')

    args = parser.parse_args()
    main(args)
