import torch
from torch import nn, optim, autograd
import numpy as np
import random
from torch.nn import functional as F
import visdom
import matplotlib.pyplot as plt

EPOCH = 10000
h_dim = 400
batch_size = 512
viz = visdom.Visdom()
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


def weights_init(m):
    if isinstance(m, nn.Linear):
        # m.weight.data.normal_(0.0, 0.02)
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0)


def generate_image(D, G, xr, epoch):
    """
    Generates and saves a plot of the true distribution, the generator, and the
    critic.
    """
    N_POINTS = 128
    RANGE = 3
    plt.clf()

    points = np.zeros((N_POINTS, N_POINTS, 2), dtype='float32')
    points[:, :, 0] = np.linspace(-RANGE, RANGE, N_POINTS)[:, None]
    points[:, :, 1] = np.linspace(-RANGE, RANGE, N_POINTS)[None, :]
    points = points.reshape((-1, 2))
    # (16384, 2)
    # print('p:', points.shape)

    # draw contour
    with torch.no_grad():
        points = torch.Tensor(points).to(device)  # [16384, 2]
        disc_map = D(points).cpu().numpy()  # [16384]
    x = y = np.linspace(-RANGE, RANGE, N_POINTS)
    cs = plt.contour(x, y, disc_map.reshape((len(x), len(y))).transpose())
    plt.clabel(cs, inline=1, fontsize=10)
    # plt.colorbar()

    # draw samples
    with torch.no_grad():
        z = torch.randn(batch_size, 2).to(device)  # [b, 2]
        samples = G(z).cpu()  # [b, 2]
    plt.scatter(xr[:, 0], xr[:, 1], c='orange', marker='.')
    plt.scatter(samples[:, 0], samples[:, 1], c='green', marker='+')

    viz.matplot(plt, win='contour', opts=dict(title='p(x):%d' % epoch))


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
    # G.apply(weights_init)
    # D.apply(weights_init)
    # print(G)
    # print(D)
    optim_G = optim.Adam(G.parameters(), lr=5e-4, betas=(0.5, 0.9))
    optim_D = optim.Adam(D.parameters(), lr=5e-4, betas=(0.5, 0.9))
    viz.line([[0, 0]], [0], win='loss', opts=dict(title='loss', legend=['D', 'G']))

    for epoch in range(EPOCH):

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
            lossg = -torch.log(1 - predg).sum()

            # aggregate all
            loss_D = (lossr + lossg) / 2 * batch_size

            # optimize
            optim_D.zero_grad()
            loss_D.backward()
            optim_D.step()

        # 2. train Generator
        z = torch.randn(batch_size, 2).to(device)
        loss_G = -torch.log(D(G(z))).mean()

        optim_G.zero_grad()
        loss_G.backward()
        optim_G.step()

        if epoch % 50 == 0:
            viz.line([[loss_D.item(), loss_G.item()]], [epoch], win='loss', update='append')
            generate_image(D, G, x.cpu(), epoch)
            print('\nEpoch: {:3}/{}, loss_G: {:.4f}, loss_D: {:.4f}'.format(epoch + 50, EPOCH, loss_G, loss_D))


if __name__ == '__main__':
    main()