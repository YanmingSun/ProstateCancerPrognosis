import Model
import torch
from torchsummary import summary


net = Model.MultimodalNetwork()
# cnn.cpu()
net.cuda()
net.eval()

summary(net, (1, 26, 640, 640))