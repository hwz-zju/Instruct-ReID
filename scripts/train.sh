#!/bin/sh
ARCH=$1
NUM_GPUs=$2
DESC=$3
SEED=0

if [[ $# -eq 4 ]]; then
  port=${4}
else
  port=23456
fi

ROOT=../../
export PYTHONPATH=$ROOT:$PYTHONPATH

GLOG_vmodule=MemcachedClient=-1 \

"""CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch--nproc_per_node ${NUM_GPUs} --master_port ${port} examples/train_joint.py -a ${ARCH} --seed ${SEED} --margin 0.3 \
	--num-instances 4 -b 128 -j 8 --warmup-step 1000 --lr 0.00001 --alpha 3 --optimizer AdamW --weight-decay 0.0005 --scheduler cosine_lr \
	--iters 24000 --port ${port}\
	--logs-dir logs/${ARCH}-${DESC} --config ./scripts/config_ablation5.yaml --data-config <your project root> + Instruct-ReID/scripts/config_attr.yaml \
	--dropout_clo 0.0 --patch_size_clo 16 --stride_size_clo 16 --patch_size_bio 16 --stride_size_bio 16 \
	--train-list None --test_task_type attr \
  --query-list <your project root> + Instruct-ReID/data/real2/datalist/query_attr.txt \
  --gallery-list <your project root> + Instruct-ReID/data/real2/datalist/gallery_attr.txt \
  --validate \
  --test_feat_type f --attn_type dual_attn --fusion_loss all --fusion_branch bio+clot --vit_type base --vit_fusion_layer 2 \
  --root <your project root> + Instruct-ReID/data/real2"""

"""CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch--nproc_per_node ${NUM_GPUs} --master_port ${port} examples/train_joint.py -a ${ARCH} --seed ${SEED} --margin 0.3 \
	--num-instances 4 -b 128 -j 8 --warmup-step 1000 --lr 0.00001 --alpha 3 --optimizer AdamW --weight-decay 0.0005 --scheduler cosine_lr \
	--iters 24000 --port ${port}\
	--logs-dir logs/${ARCH}-${DESC} --config ./scripts/config_ablation5.yaml --data-config <your project root> + Instruct-ReID/scripts/config_ltcc.yaml \
	--dropout_clo 0.0 --patch_size_clo 16 --stride_size_clo 16 --patch_size_bio 16 --stride_size_bio 16 \
	--train-list None --test_task_type cc \
  --query-list <your project root> + Instruct-ReID/data/ltcc/datalist/query.txt \
  --gallery-list <your project root> + Instruct-ReID/data/ltcc/datalist/gallery.txt \
  --validate \
  --test_feat_type f --attn_type dual_attn --fusion_loss all --fusion_branch bio+clot --vit_type base --vit_fusion_layer 2 \
  --root <your project root> + Instruct-ReID/data/ltcc"""

"""CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch--nproc_per_node ${NUM_GPUs} --master_port ${port} examples/train_joint.py -a ${ARCH} --seed ${SEED} --margin 0.3 \
	--num-instances 4 -b 128 -j 8 --warmup-step 1000 --lr 0.00001 --alpha 3 --optimizer AdamW --weight-decay 0.0005 --scheduler cosine_lr \
	--iters 24000 --port ${port}\
	--logs-dir logs/${ARCH}-${DESC} --config ./scripts/config_ablation5.yaml --data-config '<your project root> + /Instruct-ReID/scripts/config_prcc.yaml \
	--dropout_clo 0.0 --patch_size_clo 16 --stride_size_clo 16 --patch_size_bio 16 --stride_size_bio 16 \
	--train-list None --test_task_type cc \
  --query-list <your project root> + Instruct-ReID/data/prcc/datalist/query.txt \
  --gallery-list <your project root> + Instruct-ReID/data/prcc/datalist/gallery.txt \
  --validate \
  --test_feat_type f --attn_type dual_attn --fusion_loss all --fusion_branch bio+clot --vit_type base --vit_fusion_layer 2 \
  --root <your project root> + Instruct-ReID/data/prcc"""

"""CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch--nproc_per_node ${NUM_GPUs} --master_port ${port} examples/train_joint.py -a ${ARCH} --seed ${SEED} --margin 0.3 \
	--num-instances 4 -b 128 -j 8 --warmup-step 1000 --lr 0.00001 --alpha 1 --optimizer AdamW --weight-decay 0.0005 --scheduler cosine_lr \
	--iters 24000 --port ${port}\
	--logs-dir logs/${ARCH}-${DESC} --config ./scripts/config_ablation5.yaml --data-config '<your project root> + /Instruct-ReID/scripts/config_vc.yaml \
	--dropout_clo 0.0 --patch_size_clo 16 --stride_size_clo 16 --patch_size_bio 16 --stride_size_bio 16 \
	--train-list None --test_task_type cc \
  --query-list <your project root> + Instruct-ReID/data/vc_clothes/datalist/query.txt \
  --gallery-list <your project root> + Instruct-ReID/data/vc_clothes/datalist/gallery.txt \
  --validate \
  --test_feat_type f --attn_type dual_attn --fusion_loss all --fusion_branch bio+clot --vit_type base --vit_fusion_layer 2 \
  --root <your project root> + Instruct-ReID/data/vc_clothes"""

"""CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch--nproc_per_node ${NUM_GPUs} --master_port ${port} examples/train_joint.py -a ${ARCH} --seed ${SEED} --margin 0.3 \
	--num-instances 4 -b 128 -j 8 --warmup-step 1000 --lr 0.00001 --alpha 3 --optimizer AdamW --weight-decay 0.0005 --scheduler cosine_lr \
	--iters 24000 --port ${port}\
	--logs-dir logs/${ARCH}-${DESC} --config ./scripts/config_ablation5.yaml --data-config <your project root> + Instruct-ReID/scripts/config_market.yaml \
	--dropout_clo 0.0 --patch_size_clo 16 --stride_size_clo 16 --patch_size_bio 16 --stride_size_bio 16 \
	--train-list None --test_task_type sc\
  --query-list <your project root> + Instruct-ReID/data/market/datalist/query.txt \
  --gallery-list <your project root> + Instruct-ReID/data/market/datalist/gallery.txt \
  --validate \
  --test_feat_type f --attn_type dual_attn --fusion_loss all --fusion_branch bio+clot --vit_type base --vit_fusion_layer 2 \
  --root <your project root> + Instruct-ReID/data/market"""

"""CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch--nproc_per_node ${NUM_GPUs} --master_port ${port} examples/train_joint.py -a ${ARCH} --seed ${SEED} --margin 0.3 \
	--num-instances 4 -b 128 -j 8 --warmup-step 1000 --lr 0.00001 --alpha 3 --optimizer AdamW --weight-decay 0.0005 --scheduler cosine_lr \
	--iters 24000 --port ${port}\
	--logs-dir logs/${ARCH}-${DESC} --config ./scripts/config_ablation5.yaml --data-config '<your project root> + /Instruct-ReID/scripts/config_cuhk.yaml \
	--dropout_clo 0.0 --patch_size_clo 16 --stride_size_clo 16 --patch_size_bio 16 --stride_size_bio 16 \
	--train-list None --test_task_type sc\
  --query-list <your project root> + Instruct-ReID/data/cuhk/datalist/query.txt \
  --gallery-list <your project root> + Instruct-ReID/data/cuhk/datalist/gallery.txt \
  --validate \
  --test_feat_type f --attn_type dual_attn --fusion_loss all --fusion_branch bio+clot --vit_type base --vit_fusion_layer 2 \
  --root <your project root> + Instruct-ReID/data/cuhk"""

"""CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch--nproc_per_node ${NUM_GPUs} --master_port ${port} examples/train_joint.py -a ${ARCH} --seed ${SEED} --margin 0.3 \
	--num-instances 4 -b 128 -j 8 --warmup-step 1000 --lr 0.00001 --alpha 3 --optimizer AdamW --weight-decay 0.0005 --scheduler cosine_lr \
	--iters 24000 --port ${port}\
	--logs-dir logs/${ARCH}-${DESC} --config ./scripts/config_ablation5.yaml --data-config '<your project root> + /Instruct-ReID/scripts/config_msmt.yaml \
	--dropout_clo 0.0 --patch_size_clo 16 --stride_size_clo 16 --patch_size_bio 16 --stride_size_bio 16 \
	--train-list None --test_task_type sc \
  --query-list <your project root> + Instruct-ReID/data/msmt/datalist/query.txt \
  --gallery-list <your project root> + Instruct-ReID/data/msmt/datalist/gallery.txt \
  --validate \
  --test_feat_type f --attn_type dual_attn --fusion_loss all --fusion_branch bio+clot --vit_type base --vit_fusion_layer 2 \
  --root <your project root> + Instruct-ReID/data/msmt"""

"""CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch--nproc_per_node ${NUM_GPUs} --master_port ${port} examples/train_joint.py -a ${ARCH} --seed ${SEED} --margin 0.3 \
	--num-instances 4 -b 128 -j 8 --warmup-step 1000 --lr 0.00001 --alpha 3 --optimizer AdamW --weight-decay 0.0005 --scheduler cosine_lr \
	--iters 24000 --port ${port}\
	--logs-dir logs/${ARCH}-${DESC} --config ./scripts/config_ablation5.yaml --data-config '<your project root> + /Instruct-ReID/scripts/config_llcm.yaml \
	--dropout_clo 0.0 --patch_size_clo 16 --stride_size_clo 16 --patch_size_bio 16 --stride_size_bio 16 \
	--train-list None --test_task_type cross \
  --query-list <your project root> + Instruct-ReID/data/llcm/query_v2.txt \
  --gallery-list <your project root> + Instruct-ReID/data/llcm/gallery_v2.txt \
  --validate \
  --test_feat_type f --attn_type dual_attn --fusion_loss all --fusion_branch bio+clot --vit_type base --vit_fusion_layer 2 \
  --root <your project root> + Instruct-ReID/data/llcm"""

"""CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch--nproc_per_node ${NUM_GPUs} --master_port ${port} examples/train_joint.py -a ${ARCH} --seed ${SEED} --margin 0.3 \
	--num-instances 4 -b 128 -j 8 --warmup-step 1000 --lr 0.00001 --alpha 3 --optimizer AdamW --weight-decay 0.0005 --scheduler cosine_lr \
	--iters 24000 --port ${port}\
	--logs-dir logs/${ARCH}-${DESC} --config ./scripts/config_ablation5.yaml --data-config <your project root> + Instruct-ReID/scripts/config_ctcc.yaml \
	--dropout_clo 0.0 --patch_size_clo 16 --stride_size_clo 16 --patch_size_bio 16 --stride_size_bio 16 \
	--train-list None --test_task_type ctcc \
  --query-list <your project root> + Instruct-ReID/data/real2/datalist/query.txt \
  --gallery-list <your project root> + Instruct-ReID/data/real2/datalist/gallery.txt \
  --validate \
  --test_feat_type f --attn_type dual_attn --fusion_loss all --fusion_branch bio+clot --vit_type base --vit_fusion_layer 2 \
  --root <your project root> + Instruct-ReID/data/real2"""

"""CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch--nproc_per_node ${NUM_GPUs} --master_port ${port} examples/train_joint.py -a ${ARCH} --seed ${SEED} --margin 0.3 \
	--num-instances 4 -b 128 -j 8 --warmup-step 1000 --lr 0.00004 --alpha 3 --optimizer AdamW --weight-decay 0.0005 --scheduler cosine_lr \
	--iters 24000 --port ${port}\
	--logs-dir logs/${ARCH}-${DESC} --config ./scripts/config_ablation5.yaml --data-config <your project root> + Instruct-ReID/scripts/config_joint.yaml \
	--dropout_clo 0.0 --patch_size_clo 16 --stride_size_clo 16 --patch_size_bio 16 --stride_size_bio 16 \
	--train-list None --test_task_type ctcc \
  --query-list <your project root> + Instruct-ReID/data/real2/datalist/query.txt \
  --gallery-list <your project root> + Instruct-ReID/data/real2/datalist/gallery.txt \
  --validate \
  --test_feat_type f --attn_type dual_attn --fusion_loss all --fusion_branch bio+clot --vit_type base --vit_fusion_layer 2 \
  --root <your project root> + Instruct-ReID/data/real2"""