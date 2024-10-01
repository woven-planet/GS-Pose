# MIT License
# Copyright (c) 2017 Fei Xia
# Permission is granted to use, copy, modify, merge, publish, and distribute this software.
# The software is provided "as is", without warranty of any kind.
# For more details, see the full license https://opensource.org/license/MIT.

from torch.nn.modules.module import Module

from ..functions.nnd import NNDFunction

class NNDModule(Module):
    def forward(self, input1, input2):
        return NNDFunction.apply( input1, input2 )
