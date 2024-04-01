# instruct-reid

This repo provides a basic training and testing framework for instruction guided person re-identification (instruct-ReID). 

### Installation
```
Requirements:
torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0
ftfy==5.8
regex==2023.10.3 
tqdm==4.65.0
transformers==4.31.0
bytecode ==0.15.1
matplotlib==3.8.0
scikit-learn ==1.3.0
opencv-python==4.9.0.80
pyyaml==6.0.1
clip==0.2.0
timm==0.9.16
tensorboardX==2.6.2.2
easydict==1.13
chardet==5.2.0
```

### Prepare Pre-trained Models
```
shell
mkdir logs && cd logs && mkdir pretrained
```
download pretrained model [pass_vit_base_full.pth](https://drive.google.com/file/d/1sZUrabY6Lke-BJoxOEviX5ALJ017x4Ft/view) and [ALBEF.pth](https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/ALBEF.pth) to pretrained directory

download [pytorch_model.bin](https://huggingface.co/google-bert/bert-base-uncased/blob/main/pytorch_model.bin) to bert-base-uncased

The file tree should be
```
logs
└── pretrained
    └── pass_vit_base_full.pth
    └── ALBEF.pth
bert-base-uncased
└── pytorch_model.bin

```


### Prepare data
```
shell
mkdir data
```
cp the dataset and annotation datalist to data directory.

We provide OmniReID annotation datalist download [link](https://drive.google.com/file/d/1FbFhQMIs7LFdLpO_ML6Nu9UsRUr00acw/view?usp=drive_link)

We provide the images original homepage link and please download the CUHK dataset from [here](https://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html), ltcc dataset from [here](https://naiq.github.io/LTCC_Perosn_ReID.html), Market1501 dataset from [here](https://zheng-lab.cecs.anu.edu.au/Project/project_reid.html), MSMT dataset from [here](https://www.pkuvmc.com/dataset.html), PRCC dataset from [here](https://www.isee-ai.cn/~yangqize/clothing.html), COCAS+ Real1 dataset from [here](https://github.com/Chenhaobin/COCAS-plus), COCAS+ Real2 dataset from [here](https://github.com/Chenhaobin/COCAS-plus), VC-Clothes dataset from [here](https://wanfb.github.io/dataset.html), LLCM dataset from [here](https://github.com/ZYK100/LLCM), CUHK-PEDES dataset from [here](https://github.com/layumi/Image-Text-Embedding/tree/master/dataset/CUHK-PEDES-prepare) and PLIP dataset from [here](https://github.com/zplusdragon/plip).

The file tree should be
```
data
└── cuhk
    └── datalist
        └── query.txt
        └── gallery.txt
        └── train.txt
    └── cuhk03_1
└── ltcc
    └── croped_clothes
    └── datalist
        └── query_sc.txt
        └── gallery_sc.txt
        └── query_cc.txt
        └── gallery_cc.txt
        └── query_general.txt
        └── gallery_general.txt
        └── train.txt
    └── LTCC_ReID
    └── templates
    └── white_shirt.jpg
└── market
    └── datalist
        └── query.txt
        └── gallery.txt
        └── train.txt
    └── Market-1501
└── msmt
    └── datalist
        └── query.txt
        └── gallery.txt
        └── train.txt
    └── MSMT17_V1
└── prcc
    └── croped_clothes
    └── datalist
        └── query.txt
        └── gallery.txt
        └── train.txt
    └── rgb
    └── prcc_A_templates
    └── white_shirt.jpg
└── real1
    └── COCAS
    └── datalist
        └── runner_real1_v1_gpt.json
        └── train_attr.txt
        └── train_ctcc.txt
└── real2
    └── real_reid_image_face_blur
    └── datalist
        └── runner_real2_v1_gpt.json
        └── query_attr.txt
        └── gallery_attr.txt
        └── query.txt
        └── gallery.txt
└── vc_clothes
    └── croped_image
    └── datalist
        └── query.txt
        └── gallery.txt
        └── train.txt
    └── gallery
    └── query
    └── train
    └── white_shirt.jpg
└── llcm
    └── LLCM
    └── query.txt
    └── gallery.txt
    └── train.txt
└── CUHK-Pedes
    └── imgs
    └── caption_t2i_v2.json
    └── query_t2i_v2.txt
    └── gallery_t2i_v2.txt
    └── train_t2i_v2.txt
└── PLIP
    └── Part1
    └── Part2
    └── Part3
    └── Part4
    └── Part5
    └── Part6
    └── caption_t2i.json
    └── train_t2i.txt
```

### Training

```
shell
./scripts/train.sh transformer_dualattn_joint ${gpu_num} ${description} ${port}
# e.g., sh ./scripts/train.sh transformer_dualattn_joint 1 debug 6656
```

### Testing

```
shell
./scripts/test.sh transformer_dualattn_joint ${/PATH/TO/YOUR/MODEL/} ${test_task_type} ${query-txt} ${gallery-txt} ${root_path} # default 1 GPUs
# e.g., sh ./scripts/test.sh transformer_dualattn_joint <your project root>/Instruct-ReID/checkpoint_cuhk.pth.tar sc <your project root>/Instruct-ReID/data/cuhk/datalist/query.txt <your project root>+/Instruct-ReID/data/cuhk/datalist/gallery.txt <your project root>+/Instruct-ReID/data/cuhk
```

### inference model
We provide inference model for each task at [link](https://drive.google.com/drive/folders/1QBJ-GTI3Luvv-9c7_QJbhlx3ljo2t5kb?usp=drive_link).

