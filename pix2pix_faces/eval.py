from __future__ import print_function
import argparse
from model import Generator
from PIL import Image

import torch
import torchvision

import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str)
parser.add_argument('--model', type=str)

args = parser.parse_args()

checkpoint = torch.load(args.model, map_location='cpu')

# Instanciate Generator
generator = Generator(3,3)
generator.load_state_dict(checkpoint['generator'])
generator.eval()

# Load image.
inp_im = Image.open(args.image)

# Get original dimension of image.
width, height = inp_im.size

transform = transforms = torchvision.transforms.Compose([
             torchvision.transforms.Resize((286,286)),
             torchvision.transforms.ToTensor()])

# Scale and convert to Pytorch Tensor.
inp_im = transform(inp_im)

# Evaluate generator:
out_im = generator(inp_im.view(1,3,286,286))
out_im = out_im[0]

fig, ax = plt.subplots(1,2)
ax[0].imshow(inp_im.data.numpy().transpose(1,2,0))
ax[1].imshow(out_im.data.numpy().transpose(1,2,0))
plt.savefig('hola.jpg')

# Return image to PIL object.
transform = transforms = torchvision.transforms.Compose([
	         torchvision.transforms.ToPILImage(mode='RGB'),
             torchvision.transforms.Resize((height,width))])

out_im = transform(out_im)
out_im.save('pixeled.jpeg')


print('Imagen: {}'.format(args.image))
print('Imagen guardada en "pixeled.jpeg"')
