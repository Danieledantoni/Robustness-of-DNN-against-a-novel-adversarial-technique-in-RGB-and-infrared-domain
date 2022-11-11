import torch 
from torch import nn
import numpy as np
import torchvision
from torchvision.transforms import ToTensor
from scipy import ndimage
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Clean_GAN(nn.Module):
  def __init__(self, gan, model):
    super().__init__()
    self.gan = gan
    self.model = model

  def forward(self, x):
    x = self.gan(x)
    logits = self.model(x)
    return logits

class Mitigation_rand(nn.Module):
  def __init__(self, start, stop, model, vit = None):
    super().__init__()
    self.start = start
    self.stop = stop
    self.vit = vit
    self.model = model

  def forward(self, x):
    max_size = int(self.stop * 224)
    resize_layer = nn.Upsample(scale_factor = (np.random.uniform(self.start, self.stop)), mode = 'nearest')
    x = resize_layer(x)
    left = np.random.randint(10)
    top = np.random.randint(10)
    right = max_size - x.size()[2] - left
    bottom = max_size - x.size()[1] - top
    padding_layer = nn.functional.pad(x, (left, right, top, bottom))
    if self.vit:
      x = torchvision.transforms.Resize((224, 224))(x)
    logits = self.model(x)
    return logits

class Filter_detection(nn.Module):
  def __init__(self, filter, initial_window_size, max_window_size, threshold, model, window_size_median = 3):
    super().__init__()
    self.model = model
    self.filter = filter
    self.initial_window_size = initial_window_size
    self.max_window_size = max_window_size
    self.window_size_median = window_size_median
    self.threshold = threshold
    
  def forward(self, batch):
    soft = nn.Softmax(dim=1)
    image_to_keep = []
    for idx, image in enumerate(batch):
      copy = image.clone()

      if self.filter.__name__ == 'amf':
        probs = soft(self.model(image.unsqueeze(0)))
        filtered_image = self.filter(image.cpu().detach().numpy(), self.initial_window_size, self.max_window_size)
        filtered_image = ToTensor()(filtered_image)
        filtered_probs = soft(self.model(filtered_image.permute(1,2,0).unsqueeze(0).to(device)))

      if self.filter.__name__ == 'median_filter':
        probs = soft(self.model(image.unsqueeze(0)))
        filtered_image = ndimage.median_filter(image.detach().cpu(), size = 3)
        filtered_image = ToTensor()(filtered_image).permute(1,2,0)
        filtered_probs = soft(self.model(filtered_image.unsqueeze(0).to(device)))        

      norm = torch.norm((probs - filtered_probs), 1)
                
      if norm <= self.threshold:
        image_to_keep.append(idx)
        
    batch = batch[image_to_keep]
    logits = self.model(batch)
    return logits, image_to_keep