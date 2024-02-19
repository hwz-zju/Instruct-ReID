#!/bin/sh

ARCH=$1
export PATH=~/.local/bin/:$PATH

CUDA_VISIBLE_DEVICES=0 python -u examples/test_joint.py -a ${ARCH} --resume $2 --patch_size_clo 16 --stride_size_clo 16 --patch_size_bio 16 --stride_size_bio 16 -t $3 \
    --query-list $4 \
    --gallery-list $5 \
    --validate_feat fusion --config ./scripts/config_ablation5.yaml \
    --attn_type dual_attn --fusion_loss all --fusion_branch bio+clot --vit_type base --vit_fusion_layer 2 --test_feat_type f \
    --root $6
