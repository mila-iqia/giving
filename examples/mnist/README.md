
# MNIST example

This is essentially a copy of PyTorch's [basic MNIST example](https://github.com/pytorch/examples/tree/master/mnist), adapted to use `give` and which demonstrates various ways to log data to the terminal and to logging services such as Weights and Biases.

[View the diff](https://github.com/breuleux/giving/compare/mnist-original...mnist-new) between the original version and the enhanced version.

Notice how each feature is neatly isolated in its own function: for example, if you don't use `wandb`, there is no need to install or import it. Everything is orthogonal, so it is also theoretically possible to log to multiple services at the same time!

## Install

```bash
git clone git@github.com:breuleux/giving.git
cd examples/mnist
pip install -r requirements.txt
```

## Try out

You should try out the following options:

```bash
python main.py
python main.py --simple
python main.py --rich
python main.py --display-all

# Requires installing and setting up wandb
python main.py --wandb username:project

# Requires installing and setting up mlflow
python main.py --mlflow
```

## Image

![mnist-rich](https://github.com/breuleux/giving/raw/media/media/mnist-rich.gif)
