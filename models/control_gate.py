import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Control_gate(nn.Module):
    def __init__(self, gate_input_size, gate_linear_size=512, alpha=0.5, gate_mode='residual'):
        '''
        gate_input_size: strat_input embedding size
        alpha: hyper parameter for the residual mode
        gate_mode: if residual: alpha*encoder_output + (1-alpha)*gated_encoder_output
        '''
        super().__init__()                             
        self.activision = F.sigmoid
        self.gate_mode = gate_mode
        self.linear = nn.Linear(gate_input_size, gate_linear_size)
        self.alpha = alpha
        
    def forward(self, strat_input, encoder_output):
        strat_input = self.activision(self.linear(strat_input))
        gated_encoder_output = strat_input * encoder_output
        if self.gate_mode == 'residual':
            output = self.alpha * (encoder_output) + (1 - self.alpha) * gated_encoder_output
        else:
            output = gated_encoder_output


        return output