import argparse
from os import listdir

import torch
import torchvision
import torchvision.transforms as tranforms

import matplotlib.pyplot as plt
import numpy as np

from dataloader import PIX2PIXloader
from model import Generator, Discriminator, checkpoint

parser = argparse.ArgumentParser()
parser.add_argument("--ROOT", default='./', type=str)
parser.add_argument("--INPUT", default=None, type=str)
parser.add_argument("--OUTPUT", default=None, type=str)
parser.add_argument("--tr_per", default=0.8, type=float)
parser.add_argument("--model", default=None, type=str)
parser.add_argument("--max_epoch", default=0, type=float)

args = parser.parse_args()

# Definicion de funciones de error
L1 = torch.nn.L1Loss()
loss_binaryCrossEntropy = torch.nn.BCELoss()

# Definir error en generador
def generator_loss(disc_generated_output, gen_output, target):
  LAMBDA = 100
  gan_loss = loss_binaryCrossEntropy(disc_generated_output, torch.ones_like(disc_generated_output))
  
  l1_loss = L1(target, gen_output)
  
  total_gen_loss = gan_loss + (LAMBDA * l1_loss)
  
  return total_gen_loss

# Definir error en discriminador
def discriminator_loss(disc_real_output, disc_generated_output):
  real_loss = loss_binaryCrossEntropy(disc_real_output, torch.ones_like(disc_real_output))
  
  generated_loss = loss_binaryCrossEntropy(disc_generated_output, torch.zeros_like(disc_generated_output))
  
  total_disc_loss = real_loss + generated_loss
  
  return total_disc_loss

# Funcion de entrenamiento
def train(generator,
          discriminator, 
          discriminator_loss=None,
          generator_loss = None,
          gen_opt = None,
          dis_opt = None,
          device = 'cpu', 
          ini_epoch=0,
          max_epoch=1000,
          it_val=100):
  
 	for epoch in range(ini_epoch, max_epoch):
 		generator.train()
 		discriminator.train()
 		for i, (inp, targ) in enumerate(train_loader):
 			inp, targ = inp.to(device), targ.to(device)

 			output_image = generator(inp)
 			out_gen_discr = discriminator(output_image, inp)
 			out_trg_discr = discriminator(targ, inp)

 			discr_loss = discriminator_loss(out_trg_discr, out_gen_discr)
 			gen_loss = generator_loss(out_gen_discr, output_image, targ)

 			gen_opt.zero_grad()
 			dis_opt.zero_grad()

 			discr_loss.backward(retain_graph=True)
 			gen_loss.backward()

 			gen_opt.step()
 			dis_opt.step()

 		if epoch % it_val == 0:
 			checkpoint('./', generator, discriminator, epoch)
 			generator.eval()

 			# Evaluate some images.
 			fig, ax = plt.subplots(4,3)
 			inp, targ = next(iter(validation_loader))
 			inp, targ = inp.to(device), targ.to(device)

 			output_image = generator(inp)

 			for i in range(4):
 				ax[i,0].imshow(inp[i].data.cpu().numpy().transpose(1,2,0))
 				ax[i,1].imshow(output_image[i].data.cpu().numpy().transpose(1,2,0))
 				ax[i,2].imshow(targ[i].data.cpu().numpy().transpose(1,2,0))

 			fig.savefig('val_{}.png'.format(epoch))

# Direcciones a las imagenes.
ROOT = args.ROOT
INP = ROOT + '/' + args.INPUT
OUT = ROOT + '/' + args.OUTPUT

# Listar todas las imagenes en input
all_im = listdir(INP)

# Generar listas de nombres para train y validacion
np.random.shuffle(all_im)
N = len(all_im)

tr_dim = round(N * args.tr_per)
train_names = all_im[:tr_dim]
validation_names = all_im[tr_dim:]

# Generar los dataloader para validacion y training
transforms = torchvision.transforms.Compose([
             torchvision.transforms.Resize((300,300)),
             torchvision.transforms.RandomHorizontalFlip(),
             torchvision.transforms.RandomCrop(size=(286,286)),
             torchvision.transforms.ToTensor()
])

train_dataset = PIX2PIXloader(name_list=train_names, 
                              inp_dir=INP,
                              out_dir=OUT,
                              transform=transforms)

transforms = torchvision.transforms.Compose([
             torchvision.transforms.Resize((286,286)),
             torchvision.transforms.ToTensor()
])
validation_dataset = PIX2PIXloader(name_list=validation_names, 
                                   inp_dir=INP,
                                   out_dir=OUT,
                                   transform=transforms)

train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=0,
        shuffle=True,
        pin_memory=True
    )

validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=4,
        num_workers=0,
        pin_memory=True
    )


# Se emplea la GPU si esta disponible
device = 'cuda' if torch.cuda.is_available() else 'cpu'

in_channels, out_channels = 3, 3
# Instanciar generador y discriminador
generator = Generator(in_channels, out_channels)
discriminator = Discriminator(in_channels, out_channels)

# If model is specified load model.
ini_epoch = 0
if args.model:
  checkpoint_file = torch.load(args.model, map_location='cpu')
  generator.load_state_dict(checkpoint_file['generator'])
  discriminator.load_state_dict(checkpoint_file['discriminator'])
  ini_epoch = checkpoint_file['epoch']

generator.to(device)
discriminator.to(device)

# Optimizador.
gen_opt = torch.optim.Adam(generator.parameters(), lr= 0.0002, betas=(0.5, 0.999), weight_decay= 0.00001)
dis_opt = torch.optim.Adam(discriminator.parameters(), lr= 0.0002, betas=(0.5, 0.999), weight_decay= 0.00001)

train(generator, 
	  discriminator, 
      discriminator_loss=discriminator_loss,
      generator_loss=generator_loss,
      gen_opt=gen_opt,
      dis_opt=dis_opt,
      device=device, 
      ini_epoch=ini_epoch,
      max_epoch=1000,
      it_val=100)
