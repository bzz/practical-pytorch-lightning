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

## PT
To use pure PyTorch model
```
# train
python train-pt.py --train --epoch 40
# inference
python3 train-pt.py
```

# PTL
To train the same model using [PyTorch Lightning](https://github.com/williamFalcon/pytorch-lightning)
```
python3 train-pt_lightning.py

pip3 install tensorboard
tensorboard --logdir=lightning_logs
```


## Troubleshooting

In case `tensorboard` does not work, try

```
wget https://raw.githubusercontent.com/tensorflow/tensorboard/master/tensorboard/tools/diagnose_tensorboard.py
python diagnose_tensorboard.py
```

