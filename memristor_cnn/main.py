
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

import memtorch
import copy

from memtorch.mn.Module import patch_model
from memtorch.map.Parameter import naive_map
from memtorch.bh.crossbar.Program import naive_program

from conv_net import ConvNet
from model import Model

# Create new reference memristor
reference_memristor = memtorch.bh.memristor.VTEAM
reference_memristor_params = {"time_series_resolution": 1e-10}
memristor = reference_memristor(**reference_memristor_params)
memristor.plot_hysteresis_loop()

memristor_model = ConvNet()
memristor_model.load_state_dict(torch.load("model.ckpt"), strict=False)

patched_model = patch_model(copy.deepcopy(memristor_model),
                            memristor_model=reference_memristor,
                            memristor_model_params=reference_memristor_params,
                            module_parameters_to_patch=[torch.nn.Linear],
                            mapping_routine=naive_map,
                            transistor=True,
                            programming_routine=None,
                            tile_shape=(128, 128),
                            max_input_voltage=1.0,
                            ADC_resolution=8,
                            ADC_overflow_rate=0.,
                            quant_method='linear')

print("Hello world")
