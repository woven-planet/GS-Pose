# Copyright 2024 TOYOTA MOTOR CORPORATION

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from loguru import logger
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange, repeat
import hydra
from src.utils.profiler import PassThroughProfiler
from .transformer_module import LightGlueTransformer, Matching

class GsPoseModel(nn.Module):
    def __init__(self, config, profiler=None, debug=False):
        super().__init__()
        # Misc
        self.config = config
        self.profiler = profiler or PassThroughProfiler()
        self.debug = debug

        self.feature_type = self.config['feature_type']
        assert(self.feature_type in ['dino'])
        self.coarse_matching = Matching(
            self.config["matching"],
            profiler=self.profiler,
        )
        self.feature_fusion = LightGlueTransformer(self.config["dino_attention"])

    def forward(self, data):
        if self.feature_type == 'dino':
            model_feature_input = data['model_features']
            input_feature_input = data['input_features']
        data['model_feature_all'] = model_feature_input
        data['input_feature_all'] = input_feature_input
        result = self.feature_fusion(data)
        return result


if __name__=='__main__':
    pass

