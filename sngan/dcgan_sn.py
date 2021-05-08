import torch
from torch import nn
from tqdm.auto import tqdm
from spectral_normalization import SpectralNorm as SN
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
torch.manual_seed(0)


def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()


class Generator(nn.Module):

    def __init__(self, z_dim=10, im_chan=1, hidden_dim=64):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            self.make_gen_block(z_dim, hidden_dim * 4),
            self.make_gen_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=1),
            self.make_gen_block(hidden_dim * 2, hidden_dim),
            self.make_gen_block(hidden_dim, im_chan, kernel_size=4, final_layer=True)
        )

    def make_gen_block(self, input_channels, output_channels, kernel_size=3, stride=2, final_layer=False):
        # Build the neural block
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.ReLU()
            )
        else:  # Final Layer
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.Tanh()
            )

    def unsqueeze_noise(self, noise):
        return noise.view(len(noise), z_dim, 1, 1)

    def forward(self, noise):
        x = self.unsqueeze_noise(noise)
        return self.gen(x)


def get_noise(n_samples, z_dim, device='cpu'):
    return torch.randn(n_samples, z_dim, device=device)


class Discriminator(nn.Module):

    def __init__(self, im_chan=1, hidden_dim=16):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            self.make_disc_block(im_chan, hidden_dim),
            self.make_disc_block(hidden_dim, hidden_dim * 2),
            self.make_disc_block(hidden_dim * 2, 1, final_layer=True)
        )

    def make_disc_block(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):
        if not final_layer:
            return nn.Sequential(
                SN(nn.Conv2d(input_channels, output_channels, kernel_size, stride)),
                nn.LeakyReLU(negative_slope=0.2)
            )
        else:
            return nn.Sequential(
                SN(nn.Conv2d(input_channels, output_channels, kernel_size, stride))
            )

    def forward(self, image):
        disc_pred = self.disc(image)
        return disc_pred.view(len(disc_pred), -1)


def weight_inits(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        if hasattr(m, 'weight'):
            nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 0, 0.02)
        nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    criterion = nn.BCEWithLogitsLoss()
    z_dim = 64
    display_step = 1000
    batch_size = 128
    lr = 0.0002

    beta_1 = 0.5
    beta_2 = 0.999
    device = 'cuda'

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    dataloader = DataLoader(
        MNIST('../', download=False, transform=transform),
        batch_size=batch_size,
        shuffle=True
    )

    gen = Generator(z_dim).to(device)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta_1, beta_2))
    disc = Discriminator().to(device)
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr, betas=(beta_1, beta_2))
    gen = gen.apply(weight_inits)
    disc = disc.apply(weight_inits)

    n_epochs = 30
    cur_step = 0
    mean_generator_loss = 0
    mean_disc_loss = 0

    for epoch in range(n_epochs):

        for real, _ in tqdm(dataloader):

            # update discriminator
            cur_batch_size = len(real)
            real = real.to(device)

            disc_opt.zero_grad()
            fake_noise = get_noise(cur_batch_size, z_dim, device)
            fake = gen(fake_noise)
            disc_fake_pred = disc(fake.detach())
            disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
            disc_real_pred = disc(real)
            disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
            disc_loss = (disc_fake_loss + disc_real_loss) / 2

            mean_disc_loss += disc_loss.item() / display_step

            disc_loss.backward()
            disc_opt.step()

            # update generator
            gen_opt.zero_grad()
            fake_noise_2 = get_noise(cur_batch_size, z_dim, device)
            fake_2 = gen(fake_noise_2)
            disc_pred = disc(fake_2)
            gen_loss = criterion(disc_pred, torch.ones_like(disc_pred))
            gen_loss.backward()
            gen_opt.step()

            mean_generator_loss += gen_loss.item() / display_step

            if cur_step % display_step == 0 and cur_step > 0:
                print(f'Step {cur_step}: Generator loss: {mean_generator_loss}, Discriminator loss: {mean_disc_loss}')
                show_tensor_images(fake)
                show_tensor_images(real)
                mean_disc_loss = 0
                mean_generator_loss = 0
            cur_step += 1

