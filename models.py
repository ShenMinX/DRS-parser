import torch
import torch.nn as nn

class Linear_classifiers(nn.Module):
    def __init__(self , embed_size, syms_size,frgs_size, intergs_size, dropout_rate):
        super(Linear_classifiers, self).__init__()

        self.output_s = nn.Linear(embed_size, syms_size)
        self.output_f = nn.Linear(embed_size, frgs_size)
        self.output_i = nn.Linear(embed_size, intergs_size)

        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, embeded, embeded2):

        inputs = self.dropout(embeded)
        inputs2 = self.dropout(embeded2)

        output_syms = self.output_s(inputs)
        output_frgs = self.output_f(inputs2)
        output_intergs = self.output_i(inputs2)

        return output_syms, output_frgs, output_intergs
        
