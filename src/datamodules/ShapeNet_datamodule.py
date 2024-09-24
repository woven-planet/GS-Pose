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
from src.datasets.ShapeNet_dataset import ShapeNetDataset
from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader
from pathlib import Path


class ShapeNetDataModule(LightningDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.dataset_path = Path(kwargs['dataset_path'])
        self.batch_size = kwargs["batch_size"]
        self.num_workers = kwargs["num_workers"]
        self.pin_memory = kwargs["pin_memory"]
        self.train_symmetry = kwargs['enable_symmetry']
        self.train_obj_num = kwargs['train_obj_num']
        self.train_datasource = kwargs['train_datasource']
        assert(self.train_datasource in ['h6d','shapenet'])

        # Data related
        self.module_sample_num = kwargs["module_sample_num"]
        self.input_sample_num = kwargs["input_sample_num"]
        self.train_percent = kwargs["train_percent"]
        self.val_percent = kwargs["val_percent"]
        self.train_cat = kwargs['train_cat']


        # Loader parameters:
        self.train_loader_params = {
            "batch_size": self.batch_size,
            "shuffle": True,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
        }
        self.val_loader_params = {
            "batch_size": 1,
            "shuffle": False,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
        }
        self.test_loader_params = {
            "batch_size": 1,
            "shuffle": False,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
        }

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        """ Load data. Set variable: self.data_train, self.data_val, self.data_test"""
        assert(self.train_datasource=='shapenet')
        train_set = ShapeNetDataset(
            root=self.dataset_path,
            cat=self.train_cat,
            module_sample_num=self.module_sample_num,
            input_sample_num=self.input_sample_num,
            split='train',
            train_symmetry=self.train_symmetry,
            train_obj_num=self.train_obj_num,
        )
        print("=> Read train anno file: ", self.dataset_path)
        val_set = ShapeNetDataset(
            root=self.dataset_path,
            cat=self.train_cat,
            module_sample_num=self.module_sample_num,
            input_sample_num=self.input_sample_num,
            split='val',
            train_symmetry=self.train_symmetry,
            train_obj_num=self.train_obj_num,
        )

        self.data_train = train_set
        self.data_val = val_set
        self.data_test = val_set

    def train_dataloader(self):
        return DataLoader(dataset=self.data_train, **self.train_loader_params)

    def val_dataloader(self):
        return DataLoader(dataset=self.data_val, **self.val_loader_params)

    def test_dataloader(self):
        return DataLoader(dataset=self.data_test, **self.test_loader_params)
