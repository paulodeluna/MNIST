import torch, pdb
from torch.utils.data import DataLoader
from torch import nn
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

def show(tensor, ch=1,size=(28,28), num=16):
  # Transform data structure
  data = tensor.detach().cpu().view(-1,ch,*size)
  # Change the order of the dimensions
  grid = make_grid(data[:num], nrow = 4).permute(1,2,0)
  # show
  plt.imshow(grid)
  plt.show()
  
  epochs = 500
cur_step = 0
info_step = 300
mean_gen_loss = 0
mean_disc_loss = 0

z_dim = 64
lr = 0.0001
loss_func = nn.BCEWithLogitsLoss() # take inp and apply sigmund

bs = 120 # How many images we are processing
device = 'cuda' # process data in the gpu

dataloader = DataLoader(MNIST('.', download=True, transform=transforms.ToTensor()), shuffle=True, batch_size=bs)

def genBlock(inp, out):
  return nn.Sequential(
  nn.Linear(inp, out),
  nn.BatchNorm1d(out),
  nn.ReLU(inplace=True)
  )

class Generator(nn.Module):
  def __init__(self, z_dim=64, i_dim= 784, h_dim=128):
    super().__init__()
    self.gen = nn.Sequential(
    genBlock(z_dim, h_dim),
    genBlock(h_dim, h_dim*2),
    genBlock(h_dim*2, h_dim*4),
    genBlock(h_dim*4, h_dim*8),
    nn.Linear(h_dim*8, i_dim),
    nn.Sigmoid(), # 0 - 1
    )

  def forward(self, noise):
    return self.gen(noise)

def gen_noise(number, z_dim):
  return torch.randn(number, z_dim).to(device)


def discBlock(inp, out):
  return nn.Sequential(
  nn.Linear(inp, out),
  nn.LeakyReLU(0.2)
  )

class Discriminator(nn.Module):
  def __init__(self, i_dim=784, h_dim=256):
    super().__init__()
    self.disc=nn.Sequential(
        discBlock(i_dim, h_dim*4),
        discBlock(h_dim*4, h_dim*2),
        discBlock(h_dim*2, h_dim),
        nn.Linear(h_dim, 1)
    )
 
  def forward(self, image):
    return self.disc(image)

gen = Generator(z_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
disc = Discriminator().to(device)
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)

def calc_gen_loss(loss_func, gen, disc, number, z_dim):
  noise = gen_noise(number, z_dim)
  fake = gen(noise)
  pred = disc(fake)
  targets = torch.ones_like(pred)
  gen_loss = loss_func(pred, targets)

  return gen_loss

def calc_disc_loss(loss_func, gen, disc, number, real, z_dim):
  noise = gen_noise(number, z_dim)
  fake = gen(noise)
  disc_fake = disc(fake.detach())
  disc_fake_targets = torch.zeros_like(disc_fake)
  disc_fake_loss = loss_func(disc_fake, disc_fake_targets)

  disc_real = disc(real)
  disc_real_targets = torch.ones_like(disc_real)
  disc_real_loss = loss_func(disc_real, disc_real_targets)

  disc_loss = (disc_fake_loss + disc_real_loss) / 2

  return disc_loss

for epoch in range(epochs):
  for real, _ in tqdm(dataloader):
    #train disc
    disc_opt.zero_grad()
    cur_bs = len(real)

    # reshape reals into the size of the batch
    real = real.view(cur_bs, -1)
    real = real.to(device)

    # calc loss
    disc_loss = calc_disc_loss(loss_func, gen, disc, cur_bs, real, z_dim)
    disc_loss.backward(retain_graph=True)
    disc_opt.step()

    #train gen
    gen_opt.zero_grad()
    gen_loss = calc_gen_loss(loss_func, gen, disc, cur_bs, z_dim)
    gen_loss.backward(retain_graph=True)
    gen_opt.step()

    # viz
    mean_disc_loss += disc_loss.item() / info_step
    mean_gen_loss += gen_loss.item() / info_step

    if cur_step % info_step == 0 and cur_step > 0:
      fake_noise = gen_noise(cur_bs, z_dim)
      fake = gen(fake_noise)
      show(fake)
      show(real)
      print(f'{epoch}: step {cur_step} / Gen Loss: {mean_gen_loss} / Disc Loss: {mean_disc_loss}')

      mean_gen_loss, mean_disc_loss = 0,0
    cur_step += 1
