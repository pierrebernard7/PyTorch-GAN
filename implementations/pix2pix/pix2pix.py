import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys

import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from data.dataset import HDF5Dataset

from torch.utils.tensorboard import SummaryWriter

from implementations.pix2pix.models import *
from implementations.pix2pix.datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument(
    "--sample_interval", type=int, default=500, help="interval between sampling of images from generators"
)
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints")
parser.add_argument("--gpu", type=int, default=0, help="gpu index")
opt = parser.parse_args()
print(opt)
dataset_name = 'camus'

os.makedirs("images/%s" % dataset_name, exist_ok=True)
os.makedirs("saved_models/%s" % dataset_name, exist_ok=True)

cuda = True if torch.cuda.is_available() else False

# Loss functions
criterion_GAN = torch.nn.MSELoss()
criterion_pixelwise = torch.nn.L1Loss()

# Loss weight of L1 pixel-wise loss between translated image and real image
lambda_pixel = 100

# Calculate output of image discriminator (PatchGAN)
patch = (1, opt.img_height // 2 ** 4, opt.img_width // 2 ** 4)

# Initialize generator and discriminator
generator = GeneratorUNet(in_channels=1, out_channels=4)
discriminator = Discriminator(in_channels=4)

if cuda:
    torch.cuda.set_device(opt.gpu)
    print("using GPU " + str(opt.gpu) + ": " + torch.cuda.get_device_name(opt.gpu))
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    criterion_GAN.cuda()
    criterion_pixelwise.cuda()
else:
    print("using CPU")

if opt.epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load("saved_models/%s/generator_%d.pth" % (dataset_name, opt.epoch)))
    discriminator.load_state_dict(torch.load("saved_models/%s/discriminator_%d.pth" % (dataset_name, opt.epoch)))
else:
    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Configure data loader
dataset = HDF5Dataset(file_path='../../data/Camus/camus01.hdf5', dataset_key='train',
                            input_size=(opt.img_height, opt.img_width))
dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu, pin_memory=True)

val_dataset = HDF5Dataset(file_path='../../data/Camus/camus01.hdf5', dataset_key='valid',
                            input_size=(opt.img_height, opt.img_width))
val_dataloader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=1, pin_memory=True)


# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

writer = SummaryWriter()
save_folder = 'results/'+list(writer.all_writers.keys())[0]
os.makedirs(save_folder)
if len(os.listdir(save_folder)) is not 0:
    raise Exception('Directory is not empty!')

def sample_images(batches_done):
    """Saves a generated sample from the validation set"""
    imgs = next(iter(val_dataloader))
    real_A = Variable(imgs[0].type(Tensor))
    real_B = Variable(imgs[1].type(Tensor))
    fake_B = generator(real_A)
    # real_B_grid = torchvision.utils.make_grid(
    #     torch.cat((real_B.data[0][0], real_B.data[0][1], real_B.data[0][2], real_B.data[0][3]), -2)
    # )
    # fake_B_grid = torchvision.utils.make_grid(
    #     torch.cat((fake_B.data[0][0], fake_B.data[0][1], fake_B.data[0][2], fake_B.data[0][3]), -2)
    # )
    # real_B_grid = F.interpolate(real_B_grid, size=opt.img_height)
    # fake_B_grid = F.interpolate(fake_B_grid, size=opt.img_height)

    # print('\n')
    # print(real_A.expand(-1, 4, -1, -1).data.shape)
    # print(fake_B.data.shape)
    # print(real_B.data.shape)
    fake_B = F.one_hot(fake_B.argmax(1).unsqueeze(0), num_classes=4).squeeze(0).type(torch.float32).permute(0, 3, 1, 2)
    print(fake_B.shape)
    img_sample = torch.cat((real_A.data,
                            fake_B.data[0][0].unsqueeze(0).unsqueeze(0),
                            fake_B.data[0][1].unsqueeze(0).unsqueeze(0),
                            fake_B.data[0][2].unsqueeze(0).unsqueeze(0),
                            fake_B.data[0][3].unsqueeze(0).unsqueeze(0),
                            real_B.data[0][0].unsqueeze(0).unsqueeze(0),
                            real_B.data[0][1].unsqueeze(0).unsqueeze(0),
                            real_B.data[0][2].unsqueeze(0).unsqueeze(0),
                            real_B.data[0][3].unsqueeze(0).unsqueeze(0)
                            ), -2)
    save_image(img_sample, "%s/%s.png" % (save_folder, batches_done), nrow=5, normalize=True)
    img_grid = torchvision.utils.make_grid(img_sample.data)
    writer.add_image('Sample generated images (step = batches_done)', img_grid, batches_done)


# ----------
#  Training
# ----------

prev_time = time.time()

for epoch in range(opt.epoch, opt.n_epochs):
    loss_G_epoch = 0
    loss_D_epoch = 0
    for i, batch in enumerate(dataloader):

        # Model inputs
        real_A = Variable(batch[0].type(Tensor))
        real_B = Variable(batch[1].type(Tensor))

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # GAN loss
        fake_B = generator(real_A)
        pred_fake = discriminator(fake_B, real_A)
        loss_GAN = criterion_GAN(pred_fake, valid)
        # Pixel-wise loss
        loss_pixel = criterion_pixelwise(fake_B, real_B)

        # Total loss
        loss_G = loss_GAN + lambda_pixel * loss_pixel

        loss_G.backward()

        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Real loss
        pred_real = discriminator(real_B, real_A)
        loss_real = criterion_GAN(pred_real, valid)

        # Fake loss
        pred_fake = discriminator(fake_B.detach(), real_A)
        loss_fake = criterion_GAN(pred_fake, fake)

        # Total loss
        loss_D = 0.5 * (loss_real + loss_fake)

        loss_D.backward()
        optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        loss_G_epoch += loss_G.item() / len(dataloader)
        loss_D_epoch += loss_D.item() / len(dataloader)

        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f] ETA: %s"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(dataloader),
                loss_D.item(),
                loss_G.item(),
                loss_pixel.item(),
                loss_GAN.item(),
                time_left,
            )
        )

        # If at sample interval save image
        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done)

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(), "saved_models/%s/generator_%d.pth" % (dataset_name, epoch))
        torch.save(discriminator.state_dict(), "saved_models/%s/discriminator_%d.pth" % (dataset_name, epoch))

    writer.add_scalar('Loss/generator', loss_G_epoch, epoch)
    writer.add_scalar('Loss/discriminator', loss_D_epoch, epoch)
