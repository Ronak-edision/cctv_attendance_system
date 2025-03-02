import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1

class Model(nn.Module):
  def __init__(self):
    super().__init__()
    self.facenet=InceptionResnetV1(pretrained='casia-webface' ,classify=False)
    self.facenet.last_bn=torch.nn.Identity()
    self.facenet.logits=torch.nn.Identity()

  def forward(self,x):
    return self.facenet(x)
  

  