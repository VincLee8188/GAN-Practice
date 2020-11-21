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
        return output.view()


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
    for epoch in range(50000):

        # 1. train Discriminator
        for _ in range(5):
            # 1.1. train on real data
            x = next(data_iter)
            x = torch.from_numpy(x).cuda()
            predr = D(x)
            lossr = -torch.log(predr).sum()

            # 1.2. train on generated data
            z = next(data_iter)
            z = torch.from_numpy(z).cuda()
            predg = D(G(z).detach())
            lossg = -torch.log(1. - predg).sum()

        # 2. train Generator


if __name__ == '__main__':
    main()
