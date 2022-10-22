# Training script

The script allows to train simple models on 
the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset.

One can choose between an MLP or CNN for the model
and between SGD and Adam for the optimizer.
One can easily change the model architecture using
the different layers from the `simple_autograd.nn` module.

The script also includes automatically downloading the data as well as 
saving and resuming from checkpoints.


CLI Usage:
```
usage: train.py [-h] [--data_root DATA_ROOT] [--download_data] [--save_path SAVE_PATH] [--auto_continue]
                [--model {cnn,mlp}] [--batch_norm] [--lr LR] [--epochs EPOCHS] [--batch_size BATCH_SIZE]
                [--optimizer {sgd,adam}] [--weight_decay WEIGHT_DECAY] [--sgd_momentum SGD_MOMENTUM] [--sgd_nesterov]
                [--sgd_dampening SGD_DAMPENING] [--adam_betas B B] [--verbose]

optional arguments:
  -h, --help            show this help message and exit
  --data_root DATA_ROOT
                        where to store data (default: None)
  --download_data       Whether to automatically download the data if not present. (default: False)
  --save_path SAVE_PATH
                        where to save artifacts (default: None)
  --auto_continue       Continue from previous checkpoint. (default: False)
  --model {cnn,mlp}     What model to use. [cnn,mlp] (default: mlp)
  --batch_norm          Whether to use BN in model. (default: False)
  --lr LR               lr to use (default: None)
  --epochs EPOCHS       number of epochs to train for (default: None)
  --batch_size BATCH_SIZE
                        the batch size to use (default: None)
  --optimizer {sgd,adam}
                        optimizer (default: sgd)
  --weight_decay WEIGHT_DECAY
                        Weight decay (default: 0.0)
  --sgd_momentum SGD_MOMENTUM
                        For SGD: momentum value to use (default: 0.0)
  --sgd_nesterov        For SGD: use Nesterov momentum (default: False)
  --sgd_dampening SGD_DAMPENING
                        For SGD: dampening value to use (default: 0.0)
  --adam_betas B B      Betas to use. (default: [0.9, 0.999])
  --verbose             print more information (default: False)
```

# Examples commands
Training the CNN is really slow compared to the MLP 
even though it has a significantly lower
number of parameters (ironic, I know).
It takes about 5 hours per epoch for the CNN and
not even 10 seconds per epoch for the MLP.
The reason for this is the rather unoptimized convolution
operator.
However, note that the implementation works and has been
tested against torch.

The following trains an MLP using SGD:
```shell
python -m examples.train --data_root tmp/data --save_path tmp/model \
  --model mlp --lr 0.1 --optimizer sgd --batch_size 64 \
  --epoch 10 --verbose
```

In my experiments, it achieves +90% test accuracy after 2 epochs
and a final test accuracy of 95.32% in 1.5 minutes.

The following trains an MLP using Adam:
```shell
python -m examples.train --data_root tmp/data --save_path tmp/model \
  --model mlp --lr 1e-3 --optimizer adam --batch_size 64 \
  --epoch 10 --verbose
```
It achieves almost 93% test accuracy after the first epoch
and a final test accuracy of 97.32%.


The following trains a CNN using Adam:
```shell
python -m examples.train --data_root tmp/data --save_path tmp/model \
  --model cnn --lr 1e-3 --optimizer adam --batch_size 64 \
  --epoch 10 --verbose
```
Note that this takes a really long time and requires `scipy` to be installed.
