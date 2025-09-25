import os
import json
import torch
import random
import numpy as np
import torch.backends.cudnn as cudnn

def set_seed(seed: int = 0):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.benchmark = True