# Copied from PyTorch's example folder.

# Original: https://raw.githubusercontent.com/pytorch/examples/master/mnist/main.py
# Retrieved: 2021-09-01
# Modified to use the giving package

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from giving import give, given
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        give(batch_idx, train_loss=loss.item())


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            give(batch_idx)

    test_loss /= len(test_loader.dataset)
    correct /= len(test_loader.dataset)
    give(test_loss, correct)


def run(args):
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    give()
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        with give.inherit(epoch=epoch):
            with give.wrap_inherit("train", mode="train", batch_size=train_loader.batch_size, length=len(train_loader.dataset)):
                train(args, model, device, train_loader, optimizer, epoch)

            with give.wrap_inherit("test", mode="test", batch_size=test_loader.batch_size, length=len(test_loader.dataset)):
                test(model, device, test_loader)

        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


def log_terminal(gv):
    @gv.kwrap("train")
    @gv.kwrap("test")
    def _(mode):
        print(f"> Start {mode}")
        yield
        print(f"< End {mode}")

    gvtl = gv.where_any("batch_idx").throttle(0.5)
    gvtl.affix(
        progress=gvtl.kmap(
            lambda batch_idx, batch_size, length: f"{(batch_idx + 1) * batch_size}/{length}"
        )
    ).keep("mode", "epoch", "progress", "train_loss").display()

    gv.keep("test_loss", "correct").display()


def log_wandb(gv, args):
    import wandb

    entity, project = args.wandb.split(":")

    wandb.init(project=project, entity=entity, config=vars(args))

    gv["?model"].first() >> wandb.watch
    gv.keep("train_loss", "test_loss") >> wandb.log


def log_mlflow(gv, args):
    import mlflow

    mlflow.log_params(vars(args))

    gv.keep("train_loss", "test_loss") >> mlflow.log_metrics


def log_comet(gv, args):
    import comet_ml

    entity, project, api_key = args.comet.split(":")

    # Create an experiment with your api key
    experiment = comet_ml.Experiment(
        api_key=api_key,
        project_name=project,
        workspace=entity,
        auto_output_logging=False,
    )

    experiment.log_parameters(vars(args))

    gv.wrap("train", experiment.train)
    gv.wrap("test", experiment.test)

    gv["?epoch"] >> experiment.set_epoch
    gv["?batch_idx"] >> experiment.set_step

    gv.keep("train_loss", "test_loss") >> experiment.log_metrics


def log_rich(gv):
    from rich.progress import Progress
    from rich.live import Live
    from rich.table import Table
    from rich.pretty import Pretty

    rows = {}
    exclude = ["$wrap"]

    def update(values):
        for k, v in values.items():
            if isinstance(v, torch.Tensor):
                v = (v.shape, v.dtype)
            if k in exclude:
                continue
            if k in rows:
                rows[k]._object = v
            else:
                rows[k] = Pretty(v)
                table.add_row(k, rows[k])

    table = Table.grid(padding=(0, 3, 0, 0))
    table.add_column("key", style="bold green")
    table.add_column("value")

    progress = Progress(auto_refresh=False)
    task = progress.add_task("----")
    table.add_row("progress", progress)

    @gv.kwrap("train")
    @gv.kwrap("test")
    def _(epoch, length, mode):
        descr = f"Epoch #{epoch}" if mode == "train" else "Test"
        progress.reset(task, total=length, description=descr)
        yield

    gv.wrap("run", Live(table, refresh_per_second=4))

    @gv.where("batch_idx").ksubscribe
    def _(batch_size):
        progress.advance(task, batch_size)

    gv >> update


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--wandb', default="",
                        help='username:project for wandb')
    parser.add_argument('--comet', default="",
                        help='username:project for comet')
    parser.add_argument('--mlflow', action='store_true', default=False,
                        help='whether to use mlflow to store logs')
    parser.add_argument('--rich', action='store_true', default=False,
                        help='whether to show a rich display')
    args = parser.parse_args()

    with given() as gv:

        if args.rich:
            log_rich(gv)
        else:
            log_terminal(gv)

        if args.wandb:
            log_wandb(gv, args)

        if args.comet:
            log_comet(gv, args)

        if args.mlflow:
            log_mlflow(gv, args)

        give(args=vars(args))

        with give.wrap("run"):
            run(args)


if __name__ == '__main__':
    main()
