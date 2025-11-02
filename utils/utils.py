import torch

def torch_seed(seed=123):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deteministic = True
  torch.use_deteministic_algorithms = True