import torch
from .resnet import resnet50
from .vgg import vgg19_bn
from .densenet import densenet121

resnet_50 = resnet50()
vgg_19bn = vgg19_bn()
densenet_121 = densenet121()

resnet_50.load_state_dict(torch.load("./cifar10_models/state_dicts/resnet50.pt", map_location=torch.device('cpu')))
vgg_19bn.load_state_dict(torch.load("./cifar10_models/state_dicts/vgg19_bn.pt", map_location=torch.device('cpu')))
densenet_121.load_state_dict(
    torch.load("./cifar10_models/state_dicts/densenet121.pt", map_location=torch.device('cpu')))

resnet_50.eval()
vgg_19bn.eval()
densenet_121.eval()

for param in resnet_50.parameters():
    param.requires_grad = False
for param in vgg_19bn.parameters():
    param.requires_grad = False
for param in densenet_121.parameters():
    param.requires_grad = False
