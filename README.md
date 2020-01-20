# Tutorial: RNN PyTorch (PT) and PyTorch-Lightning (PTL)
> Char-level RNN for generating city names

## Architecture

![Neural Network architecture diagram](https://camo.githubusercontent.com/00874b6b3fe0fbe8fbe65ffdc506f57d3646a18a/68747470733a2f2f692e696d6775722e636f6d2f6a7a56726637662e706e67)

Variant of the official [PyTorch NLP Tutorial](https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html) for using RNN to generate city names, that is updated to use API of `torch.utils.data.DataLoader` and `torch.optim`.


## Prerequisite

```
pip install -r requirement.txt
./download_data.sh
```

## PT RNN LM
To use pure PyTorch model
```
# train
python train-pt.py --train --epoch 40

# inference
python3 train-pt.py
```

## PTL RNN LM
To train the same model using [PyTorch Lightning](https://github.com/williamFalcon/pytorch-lightning)
```
python3 train-pt_lightning.py
tensorboard --logdir=lightning_logs
```


## PTL seq2seq

seq2seq model on source code [CodeSearchNet dataset](https://github.com/github/CodeSearchNet/blob/master/README.md#downloading-data-from-s3) for predicting function names \w [sub-word tokenizer](https://medium.com/@makcedward/how-subword-helps-on-your-nlp-model-83dd1b836f46):
```
wget 'https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/java.zip'
unzip java.zip

# train
python3 seq2seq-code.py --data_dir="java"
tensorboard --logdir=lightning_logs

# inference (hardcoded input)
python3 seq2seq-code.py --infer "lightning_logs/version_<X>/checkpoints/_ckpt_epoch_<N>.ckpt"
```

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

