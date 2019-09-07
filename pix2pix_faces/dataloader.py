import torchvision
from torch.utils.data import Dataset
from PIL import Image
import random
import numpy as np

# Define loader class
class PIX2PIXloader(Dataset):
  '''
  Loader for pix2pix applications
  '''
  def __init__(self, 
               name_list=None, 
               inp_dir=None, 
               out_dir=None, 
               transform=torchvision.transforms.ToTensor()):
    
    self.transform = transform
    self.inp_name = []
    self.out_name = []
    
    for name in name_list:
      self.inp_name.append(inp_dir + '/' + name)
      self.out_name.append(out_dir + '/' + name)
    
  def __len__(self):
    return len(self.inp_name)
  
  def __getitem__(self, idx):
    inp_im = Image.open(self.inp_name[idx])
    out_im = Image.open(self.out_name[idx])
    
    seed = np.random.randint(2492759872)
      
    random.seed(seed)
    inp_im = self.transform(inp_im)
      
    random.seed(seed)
    out_im = self.transform(out_im)
    return inp_im, out_im