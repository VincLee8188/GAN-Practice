import torch
from torch import nn, optim, autograd
import numpy as np
import random

h_dim = 400
batch_size = 512
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, 2)
        )

    def forward(self, z):
        output = self.net(z)
        return output


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.net(x)
        return output.view(-1)


def data_generator():
    """
    8-gaussian mixture models
    :return:
    """

    scale = 2.
    centers = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1. / np.sqrt(2), 1. / np.sqrt(2)),
        (1. / np.sqrt(2), -1. / np.sqrt(2)),
        (-1. / np.sqrt(2), 1. / np.sqrt(2)),
        (-1. / np.sqrt(2), -1. / np.sqrt(2)),
    ]
    centers = [(scale * x, scale * y) for x, y in centers]

    while True:
        dataset = []
        for i in range(batch_size):
            point = np.random.randn(2) * 0.02
            center = random.choice(centers)
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset).astype(np.float32)
        dataset /= 1.1415
        yield dataset


def main():
    torch.manual_seed(2333)
    np.random.seed(2333)
    random.seed(2333)

    data_iter = data_generator()
    # x = next(data_iter)
    # [b, 2]
    # print(x.shape)

    G = Generator().to(device)
    D = Discriminator().to(device)
    # print(G)
    # print(D)
    optim_G = optim.Adam(G.parameters(), lr=5e-4, betas=(0.5, 0.9))
    optim_D = optim.Adam(D.parameters(), lr=5e-4, betas=(0.5, 0.9))

    for epoch in range(50000):

        # 1. train Discriminator
        for _ in range(5):
            # 1.1. train on real data
            x = next(data_iter)
            x = torch.from_numpy(x).to(device)
            predr = D(x)
            lossr = -torch.log(predr).sum()

            # 1.2. train on generated data
            z = torch.randn(batch_size, 2).to(device)
            predg = D(G(z).detach())
            lossg = -torch.log(1. - predg).sum()

            # aggregate all
            loss_D = (lossr + lossg) / 2 * batch_size

            # optimize
            optim_D.zero_grad()
            loss_D.backward()
            optim_D.step()

        # 2. train Generator
        z = torch.randn(batch_size, 2).to(device)
        loss_G = torch.log(1. - D(G(z))).mean()

        optim_G.zero_grad()
        loss_G.backward()
        optim_G.step()

        if epoch % 100 == 0:
            print(loss_D.item(), loss_G.item())


if __name__ == '__main__':
    main()
