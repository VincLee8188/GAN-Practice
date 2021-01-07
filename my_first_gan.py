import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
torch.manual_seed(0)


def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()


def get_generator_block(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.BatchNorm1d(output_dim),
        nn.ReLU(inplace=True)
    )


class Generator(nn.Module):
    def __init__(self, z_dim=10, im_dim=784, hidden_dim=128):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            get_generator_block(z_dim, hidden_dim),
            get_generator_block(hidden_dim, hidden_dim * 2),
            get_generator_block(hidden_dim * 2, hidden_dim * 4),
            get_generator_block(hidden_dim * 4, hidden_dim * 8),
            nn.Linear(hidden_dim * 8, im_dim),
            nn.Sigmoid()
        )

    def forward(self, noise):
        return self.gen(noise)

    def get_gen(self):
        return self.gen


def get_noise(n_samples, z_dim, device='cpu'):
    return torch.randn(n_samples, z_dim).to(device)


def get_discriminator_block(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.LeakyReLU(negative_slope=0.2)
    )


class Discriminator(nn.Module):
    def __init__(self, im_dim=784, hidden_dim=128):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            get_discriminator_block(im_dim, hidden_dim * 4),
            get_discriminator_block(hidden_dim * 4, hidden_dim * 2),
            get_discriminator_block(hidden_dim * 2, hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, image):
        return self.disc(image)

    def get_disc(self):
        return self.disc


criterion = nn.BCEWithLogitsLoss()
n_epochs = 200
z_dim = 64
display_step = 500
batch_size = 128
lr = 0.00001

dataloader = DataLoader(
    MNIST('.', download=False, transform=transforms.ToTensor()),
    batch_size=batch_size,
    shuffle=True,
)

device = 'cuda'

gen = Generator(z_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
disc = Discriminator().to(device)
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)


def get_disc_loss(gen, disc, criterion, real, num_images, z_dim, device):
    d_real = disc(real)
    real_truth = torch.ones_like(d_real)
    loss_real = criterion(d_real, real_truth)

    z = get_noise(num_images, z_dim, device)
    g_z = gen(z)
    d_g_z = disc(g_z.detach())
    fake_truth = torch.zeros_like(d_g_z)
    loss_fake = criterion(d_g_z, fake_truth)

    disc_loss = (loss_fake + loss_real) / 2
    return disc_loss


def get_gen_loss(gen, disc, criterion, num_images, z_dim, device):
    z = get_noise(num_images, z_dim, device)
    g_z = gen(z)
    d_g_z = disc(g_z)
    ground_truth = torch.ones_like(d_g_z)
    gen_loss = criterion(d_g_z, ground_truth)
    return gen_loss


# test modules

# def test_generator(z_dim, im_dim, hidden_dim, num_test=1000):
#     gen = Generator(z_dim, im_dim, hidden_dim)
#
#     test_input = torch.randn(num_test, z_dim)
#     test_output = gen(test_input)
#
#     assert tuple(test_output.shape) == (num_test, im_dim)
#     assert test_output.max() < 1, "Make sure to use a sigmoid"
#     assert test_output.min() > 0, "Make sure to use a sigmoid"
#     assert test_output.std() > 0.05, "Don't use batchnorm here"
#     assert test_output.std() < 0.15, "Don't use batchnorm here"
#
#
# def test_disc_reasonable(num_images=10):
#     import inspect, re
#     lines = inspect.getsource(get_disc_loss)
#     assert (re.search(r"to\(.cuda.\)", lines)) is None
#     assert (re.search(r"\.cuda\(\)", lines)) is None
#
#     z_dim = 64
#     gen = torch.zeros_like
#     disc = lambda x: x.mean(1)[:, None]  # add an axis at None
#     criterion = torch.mul
#     real = torch.ones(num_images, z_dim)
#     disc_loss = get_disc_loss(gen, disc, criterion, real, num_images, z_dim, 'cpu')
#     assert torch.all(torch.abs(disc_loss.mean() - 0.5))
#
#     gen = torch.ones_like
#     disc = nn.Linear(64, 1, bias=False)
#     real = torch.ones(num_images, 64) * 0.5
#     disc.weight.data = torch.ones_like(disc.weight.data) * 0.5
#     disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)
#     criterion = lambda x, y: torch.sum(x) + torch.sum(y)
#     disc_loss = get_disc_loss(gen, disc, criterion, real, num_images, z_dim, 'cpu').mean()
#     disc_loss.backward()
#     assert torch.isclose(torch.abs(disc.weight.grad.mean() - 11.25), torch.tensor(3.75))
#
#
# def test_disc_loss(max_tests=10):
#     z_dim = 64
#     gen = Generator(z_dim).to(device)
#     gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
#     disc = Discriminator().to(device)
#     disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)
#     num_steps = 0
#     for real, _ in dataloader:
#         cur_batch_size = len(real)
#         real = real.view(cur_batch_size, -1).to(device)
#
#         ### Update discriminator ###
#         # Zero out the gradient before backpropagation
#         disc_opt.zero_grad()
#
#         # Calculate discriminator loss
#         disc_loss = get_disc_loss(gen, disc, criterion, real, cur_batch_size, z_dim, device)
#         assert (disc_loss - 0.68).abs() < 0.05
#
#         # Update gradients
#         disc_loss.backward(retain_graph=True)
#
#         # Check that they detached correctly
#         assert gen.gen[0][0].weight.grad is None
#
#         # Update optimizer
#         old_weight = disc.disc[0][0].weight.data.clone()
#         disc_opt.step()
#         new_weight = disc.disc[0][0].weight.data
#
#         # Check that some discriminator weights changed
#         assert not torch.all(torch.eq(old_weight, new_weight))
#         num_steps += 1
#         if num_steps >= max_tests:
#             break


if __name__ == '__main__':
    cur_step = 0
    mean_generator_loss = 0
    mean_discriminator_loss = 0
    test_generator = True
    gen_loss = False
    error = False

    for epoch in range(n_epochs):

        for real, _ in tqdm(dataloader):
            cur_batch_size = len(real)
            real = real.view(cur_batch_size, -1).to(device)

            disc_opt.zero_grad()
            disc_loss = get_disc_loss(gen, disc, criterion, real, cur_batch_size, z_dim, device)
            disc_loss.backward(retain_graph=True)
            disc_opt.step()

            if test_generator:
                old_generator_weights = gen.gen[0][0].weight.detach().clone()

            gen_opt.zero_grad()
            gen_loss = get_gen_loss(gen, disc, criterion, cur_batch_size, z_dim, device)
            gen_loss.backward()
            gen_opt.step()

            if test_generator:
                try:
                    assert lr > 2e-7 or (gen.gen[0][0].weight.grad.abs().max() < 0.0005 and epoch == 0)
                    assert torch.any(gen.gen[0][0].weight.detach().clone() != old_generator_weights)
                except:
                    error = True
                    print("Runtime tests have failed")

            mean_discriminator_loss += disc_loss.item() / display_step
            mean_generator_loss += gen_loss.item() / display_step

            if cur_step % display_step == 0 and cur_step > 0:
                print(f'Epoch {epoch}, step {cur_step}: Generator loss: {mean_generator_loss}, '
                      f'discriminator loss:  {mean_discriminator_loss}')
                fake_noise = get_noise(cur_batch_size, z_dim, device)
                fake = gen(fake_noise)
                show_tensor_images(fake)
                show_tensor_images(real)
                mean_generator_loss = 0
                mean_generator_loss = 0
            cur_step += 1
