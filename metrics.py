# coding=utf-8
"""Helpers for computing ML metrics"""

import collections
import math

import numpy as np
import torch

@staticmethod
def acc_cm(logits, labels):
    """Calculates all confusion matrix based metrics."""
    preds = torch.argmax(logits, dim=1)
    acc = (labels == preds).float().mean()

    clss = logits.size(1)
    cm = torch.zeros((clss, clss), device=labels.device)
    for label, pred in zip(labels, preds):
        cm[label.long(), pred.long()] += 1

    tp = cm.diagonal()[1:].sum()
    fp = cm[:, 1:].sum() - tp
    fn = cm[1:, :].sum() - tp
    return (acc, tp, fp, fn)

@staticmethod
def f1_score(tp, fp, fn):
    """Calculates F1, Precision and Recall"""
    prec_rec_f1 = {}
    prec_rec_f1['precision'] = tp / (tp + fp)
    prec_rec_f1['recall'] = tp / (tp + fn)
    prec_rec_f1['f1_score'] = 2 * (
        prec_rec_f1['precision'] * prec_rec_f1['recall']) / (
            prec_rec_f1['precision'] + prec_rec_f1['recall'])
    return prec_rec_f1


def _get_ngrams(segment, max_order):
  """Extracts all n-grams up to a given maximum order from an input segment.

  Args:
    segment: text segment from which n-grams will be extracted.
    max_order: maximum length in tokens of the n-grams returned by this
        methods.

  Returns:
    The Counter containing all n-grams up to max_order in segment
    with a count of how many times each n-gram occurred.

  Copyright 2019 The Tensor2Tensor Authors.
  Licensed under the Apache License, Version 2.0
  https://github.com/tensorflow/tensor2tensor/blob/f9f63c3f3c5aab8ed8070f10662ad8012c09c17b/tensor2tensor/utils/bleu_hook.py#L40-L57
  """
  ngram_counts = collections.Counter()
  for order in range(1, max_order + 1):
    for i in range(0, len(segment) - order + 1):
      ngram = tuple(segment[i:i + order])
      ngram_counts[ngram] += 1
  return ngram_counts


def compute_bleu(reference_corpus,
                 translation_corpus,
                 max_order=4,
                 use_bp=True):
  """Computes BLEU score of translated segments against one or more references.

  Args:
    reference_corpus: list of references for each translation. Each
        reference should be tokenized into a list of tokens.
    translation_corpus: list of translations to score. Each translation
        should be tokenized into a list of tokens.
    max_order: Maximum n-gram order to use when computing BLEU score.
    use_bp: boolean, whether to apply brevity penalty.

  Returns:
    BLEU score.

  Copyright 2019 The Tensor2Tensor Authors.
  Licensed under the Apache License, Version 2.0
  https://github.com/tensorflow/tensor2tensor/blob/f9f63c3f3c5aab8ed8070f10662ad8012c09c17b/tensor2tensor/utils/bleu_hook.py#L60-129
  """
  reference_length = 0
  translation_length = 0
  bp = 1.0
  geo_mean = 0

  matches_by_order = [0] * max_order
  possible_matches_by_order = [0] * max_order
  precisions = []

  for (references, translations) in zip(reference_corpus, translation_corpus):
    reference_length += len(references)
    translation_length += len(translations)
    ref_ngram_counts = _get_ngrams(references, max_order)
    translation_ngram_counts = _get_ngrams(translations, max_order)

    overlap = dict((ngram,
                    min(count, translation_ngram_counts[ngram]))
                   for ngram, count in ref_ngram_counts.items())

    for ngram in overlap:
      matches_by_order[len(ngram) - 1] += overlap[ngram]
    for ngram in translation_ngram_counts:
      possible_matches_by_order[len(ngram)-1] += translation_ngram_counts[ngram]
  precisions = [0] * max_order
  smooth = 1.0
  for i in range(0, max_order):
    if possible_matches_by_order[i] > 0:
      precisions[i] = matches_by_order[i] / possible_matches_by_order[i]
      if matches_by_order[i] > 0:
        precisions[i] = matches_by_order[i] / possible_matches_by_order[i]
      else:
        smooth *= 2
        precisions[i] = 1.0 / (smooth * possible_matches_by_order[i])
    else:
      precisions[i] = 0.0

  if max(precisions) > 0:
    p_log_sum = sum(math.log(p) for p in precisions if p)
    geo_mean = math.exp(p_log_sum/max_order)

  if use_bp:
    if not reference_length:
      bp = 1.0
    else:
      ratio = translation_length / reference_length
      if ratio <= 0.0:
        bp = 0.0
      elif ratio >= 1.0:
        bp = 1.0
      else:
        bp = math.exp(1 - 1. / ratio)
  bleu = geo_mean * bp
  return np.float32(bleu)
