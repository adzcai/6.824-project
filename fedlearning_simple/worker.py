import argparse

import rpyc

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset

from utils import Net, get_mnist_data


def train(model, device, train_loader, optimizer, conn):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        grads = []
        for param in model.parameters():
            grads.append(param.grad.cpu().numpy())
        conn.root.send_gradient(grads)
        if batch_idx % 5 == 4:
            get_model_weights_from_server(model, conn)


def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        f"Epoch {epoch} -",
        '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)
        )
    )


def get_model_weights_from_server(model, conn):
    weights = conn.root.get_model_params()
    with torch.no_grad():
        for param, weight in zip(model.parameters(), weights):
            param.data = torch.as_tensor(np.array(weight), device=param.device)





if __name__ == "__main__":
    conn = rpyc.connect("localhost", port=12345, config={"allow_pickle": True})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("--partition_id", dest="partiton_id", type=int, required=True)
    args = parser.parse_args()

    X_train, y_train = get_mnist_data(partition=args.partiton_id, split="train")
    X_test, y_test = get_mnist_data(partition=args.partiton_id, split="test")

    train_dataset = TensorDataset(
        torch.as_tensor(X_train / 128 - 1, dtype=torch.float32),
        torch.as_tensor(y_train, dtype=torch.long)
    )
    test_dataset = TensorDataset(
        torch.as_tensor(X_test / 128 - 1, dtype=torch.float32),
        torch.as_tensor(y_test, dtype=torch.long)
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=128)
    test_loader = torch.utils.data.DataLoader(test_dataset,shuffle=False, batch_size=128)


    model = Net()
    get_model_weights_from_server(model, conn)
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0)

    epoch = 1
    while True:
        train(model, device, train_loader, optimizer, conn)
        test(model, device, test_loader, epoch)
        epoch += 1
