# GS-Pose: Category-Level Object Pose Estimation via Geometric and Semantic Correspondence 

## Overview
![teaser](assets/teaser.png "")

The paper and github pages are available:
- **arXiv**: https://arxiv.org/abs/2311.13777
- **Github Pages**: https://woven-planet.github.io/GS-Pose/

## Environment Installation
~~~
conda create -n gspose python=3.9
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
cd third_party/nndistance
python build.py install  
~~~

## Dataset Preparation

### training dataset preparation
Download the ShapeNet (https://shapenet.org/) and change the path in tools/shapenet scripts correspondingly.
Then run 
~~~
python tools/shapenet/1_shapenet_preprocess.py 
python tools/shapenet/2_shapenet_rendering.py
~~~
The data after processing should have the structure
~~~
--data--shapenet--bottle--dino_3d.pkl
                       |--color--0.png
                       |--depth--0.png
                       |--feature--0.pt
                       |--intrin_ba--0.txt
                       |--models
                       |--poses_ba--0.txt
                       |--visibility--0.txt
~~~

### testing dataset processing
Download the NOCS dataset (https://github.com/hughw19/NOCS_CVPR2019) and run 
~~~
python tools/nocs/preprocess.py
~~~

## Training and Testing 
~~~
python train_gspose.py +experiment=train.yaml ++train_cat='camera'
python train_gspose.py +experiment=test.yaml ++train_cat='camera'
~~~

## Third Party 
### nndistance
we modify the chamfer distance function  in third_party/nndistance from https://github.com/fxia22/pointGAN/tree/master/nndistance.
### transformer backbone 
we modify the transformer backbone in src/models/transformer_module from the https://github.com/cvg/LightGlue repo.

## Ackledgement
We thank the respective authors of
* https://github.com/fxia22/pointGAN/tree/master/nndistance
* https://github.com/cvg/LightGlue, https://github.com/zju3dv/OnePose_Plus_Plus
* https://github.com/facebookresearch/dinov2
* https://github.com/qq456cvb/CPPF

for their open source code.

