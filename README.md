# Tutorial: PyTorch-Lightning for research on vector representation of source code.


## Prerequisite

```
pip install -r requirement.txt
./download_data.sh
```

## RNN using PyTorch (PT) and PyTorch-Lightning (PTL)
> Char-level RNN for generating city names

This is just an example, to get the feeling of pure PT vs PTL for simple model.

### Architecture

![Neural Network architecture diagram](https://camo.githubusercontent.com/00874b6b3fe0fbe8fbe65ffdc506f57d3646a18a/68747470733a2f2f692e696d6775722e636f6d2f6a7a56726637662e706e67)

Variant of the official [PyTorch NLP Tutorial](https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html) for using RNN to generate city names, that is updated to use API of `torch.utils.data.DataLoader` and `torch.optim`.

### PT RNN LM
To use pure PyTorch model
```
# train
python train-pt.py --train --epoch 40

# inference
python3 train-pt.py
```

### PTL RNN LM
To train the same model using [PyTorch Lightning](https://github.com/williamFalcon/pytorch-lightning)
```
python3 train-pt_lightning.py
tensorboard --logdir=lightning_logs
```


## seq2seq

seq2seq baseline model on source code [CodeSearchNet dataset](https://github.com/github/CodeSearchNet/blob/master/README.md#downloading-data-from-s3) for predicting function names \w [sub-word tokenizer](https://medium.com/@makcedward/how-subword-helps-on-your-nlp-model-83dd1b836f46):
```
wget 'https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/java.zip'
unzip java.zip

# train
python3 seq2seq-code.py --data_dir="java"
tensorboard --logdir=lightning_logs

# inference (hardcoded input)
python3 seq2seq-code.py --infer --model "lightning_logs/version_<X>/checkpoints/_ckpt_epoch_<N>.ckpt"

# run evaluation on test split
python3 seq2seq-code.py --test --model "lightning_logs/version_<X>/checkpoints/_ckpt_epoch_<N>.ckpt"
```

### Evaluation

1. code2seq [Uri Alon et al. 2018](https://arxiv.org/abs/1808.01400), [ICLR 2019 poster](https://postersession.ai/poster/code2seq-generating-sequences-from-struc/)

     *Java-small* results on "code summarization" task

      model       | Prec | Rec | F1
      ------------|------|-----|----
      code2vec    |18.51 |18.74|18.62
      transformer |38.13 |26.70|31.41
      ConvAtt     |50.25 |24.62|33.05
      RNN         |42.63 |29.97|35.20
      TreeLSTM    |40.02 |31.84|35.46
      code2seq    |50.64 |37.40|43.02

   - F1, Prec, Recall
     TODO: eval using their target word-level vocabulary

2. "Structured Neural Summarization" [Patrick Fernandes et al. 2018](https://arxiv.org/abs/1811.01824)
    [poster](https://postersession.ai/poster/structured-neural-summarization/)

     *Java-small* results on "Method Naming" task

      model       |params| size |  F1  | ROUGE-2 | ROUGE-L
      ------------|------|------|------|---------|---------
      transformer | 169M |      | 24.9 | 8.3     | 27.4
      RNN         | 134M |      | 35.8 | 17.9    | 39.7
      code2seq    |  37M |137Mb*| 43   | _TODO_  | _TODO_
      GNN         |   ?  |      | 44.7 | 21.1    | 43.1

      _* size of the stripped, inference-only model_

    - F1
    - ROGUE-1/2/l
      * using a wrapper [pyrouge](https://pypi.org/project/pyrouge/) [example 1](https://github.com/CoderPat/structured-neural-summarization/blob/fef0d75bdd6142c33fddeeb0141b77d90f3423bf/rouge_evaluator.py#L86), [example 2](https://github.com/CoderPat/structured-neural-summarization/blob/master/rouge_evaluator.py#L86)
      * using a pure python [rouge](https://pypi.org/project/rouge/), [example](https://github.com/CoderPat/structured-neural-summarization/blob/fef0d75bdd6142c33fddeeb0141b77d90f3423bf/train_and_eval.py#L537)

## Transformer
TBD

## code2vec
TBD

## GNN
TBD


## Troubleshooting

In case `tensorboard` does not work, try

```
pip uninstall tb-nightly tensorboard tensorflow tensorflow-estimator
pip install tensorboard
```

and if that does not solve the issue, do

```
wget https://raw.githubusercontent.com/tensorflow/tensorboard/master/tensorboard/tools/diagnose_tensorboard.py
python diagnose_tensorboard.py
```

