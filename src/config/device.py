from torch import device, cuda

DEVICE = device('cuda') if cuda.is_available() else device('cpu')
