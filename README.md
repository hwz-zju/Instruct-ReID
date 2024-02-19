# instruct-reid

This repo provides a basic training and testing framework for instruction guided person re-identification (instruct-ReID). 

### Installation
```
Requirements:
```
ftfy
regex
tqdm
torch
torchvision
socket
sklearn
opencv
```

### Prepare Pre-trained Models
```shell
mkdir logs && cd logs && mkdir pretrained
download pretrained model pass_vit_base_full.pth and ALBEF.pth to pretrained directory
```

```
download bert-base-uncased-pytorch_model.bin and pytorch_model.bin to bert-base-uncased

```

The file tree should be
```
logs
└── pretrained
    └── pass_vit_base_full.pth
    └── ALBEF.pth
bert-base-uncased
└── bert-base-uncased-pytorch_model.bin
└── pytorch_model.bin

```


### Prepare data
```shell
mkdir data
cp the dataset and annotation datalist to data directory.
We provide OmniReID annotation datalist download [link](https://github.com/xxx)
```
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
```

### Training

```shell
./scripts/market/train.sh transformer_dualattn_joint ${gpu_num} ${description} ${port}
```

### Testing

```shell
./scripts/test.sh transformer_dualattn_joint ${/PATH/TO/YOUR/MODEL/} ${test_task_type} ${query-txt} ${gallery-txt} ${root_path} # default 1 GPUs
```

### inference model
We provide inference model for each task at [link](https://github.com/xxx).

