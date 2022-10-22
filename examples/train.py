"""
Simple training script.
"""
import argparse
import pdb
import warnings
from pathlib import Path

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    warnings.warn("`tqdm` not found! Progress bars will not work!")
    tqdm = lambda x: x

import simple_autograd.nn as nn
import simple_autograd.nn.data as data
import simple_autograd.nn.optim as optim

# (!) local import
import models


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--data_root", type=Path, help="where to store data")
    parser.add_argument("--download_data", action="store_true", default=False,
                        help="Whether to automatically download the data if not present.")
    parser.add_argument("--save_path", type=Path, help="where to save artifacts")
    parser.add_argument("--auto_continue", action="store_true", default=False, help="Continue from previous checkpoint.")

    parser.add_argument("--model", default="mlp", choices=["cnn", "mlp", "vit"], help="What model to use. [cnn,mlp,vit]")
    parser.add_argument("--batch_norm", action="store_true", default=False, help="Whether to use BN in model.")

    parser.add_argument("--lr", type=float, help="lr to use")
    parser.add_argument("--epochs", type=int, help="number of epochs to train for")
    parser.add_argument("--batch_size", type=int, help="the batch size to use")

    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam"], help="optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--sgd_momentum", type=float, default=0.0, help="For SGD: momentum value to use")
    parser.add_argument("--sgd_nesterov", action="store_true", default=False, help="For SGD: use Nesterov momentum")
    parser.add_argument("--sgd_dampening", type=float, default=0.0, help="For SGD: dampening value to use")
    parser.add_argument("--adam_betas", type=float, nargs=2, default=[0.9, 0.999], metavar="B", help="Betas to use.")

    parser.add_argument("--verbose", action="store_true", default=False, help="print more information")

    return parser


def start_training(args):
    model = get_model(args)
    if args.verbose:
        print(model)
        print(f"#Params = {nn.count_parameters(model):,}")

    train_dataset, test_dataset = get_datasets(args)
    optimizer = get_optimizer(args, model)
    criterion = nn.CrossEntropyLoss()

    if args.auto_continue:
        print("Trying to auto continue...")
        epoch = -1
        to_load = None
        for ckpt in args.save_path.glob("checkpoint-*.npz"):
            ckpt_epoch = int(ckpt.name[len("checkpoint-")].split(".")[0])
            if ckpt_epoch > epoch:
                epoch = ckpt_epoch
                to_load = ckpt

        if epoch > -1:
            print(f"Found previous checkpoint '{to_load}! Loading...")
            load_checkpoint(to_load, model, optimizer)
            epoch = epoch + 1  # to start training for next epoch
        else:
            print("No previous checkpoint found.")
            epoch = 0
    else:
        epoch = 0

    print("Starting training")
    for epoch in range(epoch, args.epochs):
        train_single_epoch(epoch, args.batch_size, criterion, model, optimizer, train_dataset)
        evaluate(epoch, args.batch_size, model, test_dataset)
        save_checkpoint(epoch, args.save_path, model, optimizer)

    print("Done with training. Woohoo!")


def train_single_epoch(epoch: int, batch_size: int, criterion: nn.Module,
                       model: nn.Module, optimizer: optim.Optimizer, train_dataset: data.MNIST):
    optimizer.zero_grad()
    total_batches = int(np.ceil(len(train_dataset) / batch_size))
    for imgs, lbls in (pbar := tqdm(train_dataset.as_batched(batch_size), total=total_batches, desc=f"Epoch [{epoch}]")):
        outputs = model(imgs)
        loss = criterion(outputs, targets=lbls)
        acc = np.equal(outputs.view(np.ndarray).argmax(1), lbls)

        if hasattr(pbar, "set_postfix"):  # in case no tqdm
            pbar.set_postfix({
                "loss": f"{loss.item():.03f}",
                "acc": f"{(acc.sum() / len(acc)).item():.03%}",
            })

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def evaluate(epoch: int, batch_size: int, model: nn.Module, test_dataset: data.MNIST):
    print()
    print()
    total_batches = int(np.ceil(len(test_dataset) / batch_size))
    total_correct = 0
    for imgs, lbls in tqdm(test_dataset.as_batched(batch_size), total=total_batches, desc=f"Eval Epoch [{epoch}]"):
        outputs = model(imgs)
        correct = np.equal(outputs.view(np.ndarray).argmax(1), lbls)

        total_correct += correct.sum().item()

    print(f"Accuracy on epoch {epoch} is {total_correct / len(test_dataset):.03%}")
    print()
    print()


def save_checkpoint(epoch: int, save_path: Path, model: nn.Module, optimizer: optim.Optimizer, last_k: int = 3):
    model_dict = model.state_dict()
    model_dict = {f"model.{key}": val for key, val in model_dict.items()}
    optim_dict = optimizer.state_dict()
    optim_dict = {f"optimizer.{key}": val for key, val in optim_dict.items()}

    save_path.mkdir(exist_ok=True)
    file_path = save_path / f"checkpoint-{epoch}.npz"
    np.savez(file_path, **model_dict, **optim_dict)
    print("Saved checkpoint to", file_path)

    for ckpt in save_path.glob("checkpoint-*.npz"):
        if int(ckpt.name[len("checkpoint-")].split(".")[0]) < epoch - last_k:
            ckpt.unlink(missing_ok=True)


def load_checkpoint(checkpoint_file: Path, model: nn.Module, optimizer: optim.Optimizer):
    assert checkpoint_file.exists()
    with np.load(str(checkpoint_file)) as d:
        i = len("model.")
        model_dict = {key[i:]: d[key] for key in d.keys() if key.startswith("model.")}
        i = len("optimizer.")
        optimizer_dict = {key[i:]: d[key] for key in d.keys() if key.startswith("optimizer.")}

    model.load_state_dict(model_dict)
    optimizer.load_state_dict(optimizer_dict)


def get_model(args):
    model_name = args.model.lower()

    if model_name == "cnn":
        norm_layer = None
        if args.batch_norm:
            norm_layer = nn.BatchNorm2d
        model = models.CNN(input_channels=1, hidden_channels=[16, 32], kernel_sizes=[3, 3], max_pool_ks=[2, 2],
                           output_classes=10, norm_layer=norm_layer)
    elif model_name == "mlp":
        norm_layer = None
        if args.batch_norm:
            norm_layer = nn.BatchNorm1d
        model = models.MLP(sizes=[28 * 28, 128, 64, 10], flatten_first=True, norm_layer=norm_layer)
    elif model_name == "vit":
        model = models.ViT(img_size=28, in_channels=1, patch_size=7, num_classes=10,
                           emb_dim=32, num_heads=4, num_blocks=1)
    else:
        raise ValueError(f"Unknown model name: '{model_name}'")

    return model


def get_optimizer(args, model: nn.Module):
    opt = None
    if args.optimizer == "sgd":
        opt = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.sgd_momentum,
            nesterov=args.sgd_nesterov,
            dampening=args.sgd_dampening,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "adam":
        opt = optim.Adam(
            model.parameters(),
            lr=args.lr,
            betas=args.adam_betas,
            weight_decay=args.weight_decay,
        )

    return opt


def get_datasets(args):
    download = args.download_data
    train_dataset = data.MNIST(data_root=args.data_root, train=True, download=download)
    test_dataset = data.MNIST(data_root=args.data_root, train=False, download=download)

    return train_dataset, test_dataset


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    start_training(args)
