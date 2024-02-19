from __future__ import absolute_import

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from typing import Optional
from torch import Tensor
import copy
from reid.models.backbone.pass_vit import vit_base_patch16_224_TransReID, Block, vit_small_patch16_224_TransReID
from reid.models.backbone.vit_albef import VisionTransformer
# from reid.models.backbone.swin_transformer import swin_base_patch4_window7_224, swin_small_patch4_window7_224, swin_tiny_patch4_window7_224
from reid.models.layers.metric import build_metric
from sklearn.cluster import KMeans
import numpy as np
import clip
from transformers import CLIPModel, CLIPProcessor
from reid.models.tokenization_bert import BertTokenizer
from reid.models.xbert import BertConfig, BertForMaskedLM
from functools import partial
import random

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
            
def shuffle_unit(features, shift, group, begin=1):

    batchsize = features.size(0)
    dim = features.size(-1)
    # Shift Operation
    feature_random = torch.cat([features[:, begin-1+shift:], features[:, begin:begin-1+shift]], dim=1)
    x = feature_random
    # Patch Shuffle Operation
    try:
        x = x.view(batchsize, group, -1, dim)
    except:
        x = torch.cat([x, x[:, -2:-1, :]], dim=1)
        x = x.view(batchsize, group, -1, dim)

    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, dim)

    return x

class MaskModule(nn.Module):
    def __init__(self, in_channels, num_masks=20):
        super(MaskModule, self).__init__()
        self.in_channels = in_channels
        self.num_masks = num_masks

        self.mask_module = nn.Sequential(
            nn.Conv2d(self.in_channels, out_channels=512, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=self.num_masks, kernel_size=1)
        )

        for sub_module in self.mask_module.modules():
            if isinstance(sub_module, nn.Conv2d):
                nn.init.xavier_normal(sub_module.weight.data)
                sub_module.bias.data.fill_(0)

    def softmax_mask(self, x):
        w = x.shape[2]
        h = x.shape[3]
        x = torch.exp(x)
        sum = torch.sum(x, dim=(2, 3), keepdim=True)
        sum = sum.repeat([1,1,w,h])
        x = x / sum
        return x

    def forward(self, x):
        # mask_feat: BxKxHxW
        mask_feat = self.mask_module(x)
        b, c, h, w = mask_feat.shape
        mask_feat = mask_feat.view(b, c, h*w)
        mask = torch.softmax(mask_feat, dim=-1).view(b, c, h, w)
        # mask = self.softmax_mask(mask_feat)
        # feat
        mask_extend = mask.unsqueeze(1)
        mask = F.max_pool3d(mask_extend, [self.num_masks, 1, 1])
        mask = mask.squeeze(1)
        feat = x
        feat = feat.mul(mask)
        return feat, mask

class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", arch=False, net_depth=9,
                 deep_prompt=False, deep_prompt_config={}):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        self.deep_prompt_embedding = None
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.arch = arch
        self.net_depth = net_depth

        self._reset_parameters()

    def _reset_parameters(self):
        if self.arch == 'deepnorm':
            for param_name, p in self.named_parameters():
                if p.dim() > 1:
                    if 'v_proj' in param_name or 'out_proj' in param_name:
                        nn.init.xavier_normal_(p, gain=(12 * self.net_depth) ** (- 0.25))
                    elif 'q_proj' in param_name or 'k_proj' in param_name:
                        nn.init.xavier_normal_(p, gain=1)
                    else:
                        nn.init.xavier_uniform_(p)
        elif self.arch == 'fan_in':
            for p in self.parameters():
                if p.dim() > 1:
                    assert p.dim() == 2
                    fan_in = p.size(1)
                    std = 1 / math.sqrt(fan_in)
                    with torch.no_grad():
                        p.normal_(0, std)
        else:
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory, # bkb_query, bs, C
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        if self.deep_prompt_embedding is not None:
            B = memory.shape[1]
            deep_prompt_embedding = self.deep_prompt_embedding(B, batch_first=False)
            tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                       key=torch.cat([deep_prompt_embedding,self.with_pos_embed(memory, pos)]),
                                       value=torch.cat([deep_prompt_embedding,memory]), attn_mask=memory_mask,
                                       key_padding_mask=memory_key_padding_mask)[0]
        else:
            tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        # if self.arch == 'pre_norm':
        #     return self.forward_pre(tgt, memory, memory_mask,
        #                             memory_key_padding_mask, pos, query_pos)
        # elif self.arch == 'deepnorm':
        #     raise NotImplementedError
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)

class Transformer_local(nn.Module):
    def __init__(self, num_classes, net_config):
        super(Transformer_local, self).__init__()
        self.base = vit_base_patch16_224_TransReID()
        attn_drop_rate = net_config.dropout_clo
        
        self.clothes = vit_base_patch16_224_TransReID(img_size=(128, 128),attn_drop_rate=attn_drop_rate)
        self.fusion = nn.Sequential(copy.deepcopy(self.base.stages[-2]), copy.deepcopy(self.base.stages[-1]))
        self.weight_block = copy.deepcopy(self.base.stages[-1])

        self.num_features = 768
        self.mask_module = MaskModule(self.num_features)
        self.norm = nn.LayerNorm(self.num_features, eps=1e-6)

        self.feat_bn = nn.BatchNorm1d(self.num_features)
        self.feat_bn.bias.requires_grad_(False)
        init.constant_(self.feat_bn.weight, 1)
        init.constant_(self.feat_bn.bias, 0)

        self.fusion_feat_bn = nn.BatchNorm1d(self.num_features)
        self.fusion_feat_bn.bias.requires_grad_(False)
        init.constant_(self.fusion_feat_bn.weight, 1)
        init.constant_(self.fusion_feat_bn.bias, 0)

        self.classifier = nn.Linear(self.num_features, num_classes, bias=False)
        init.normal_(self.classifier.weight, std=0.001)

        self.classifier_f = nn.Linear(self.num_features, num_classes, bias=False)
        init.normal_(self.classifier_f.weight, std=0.001)
    # label is unused if self.cos_layer == 'no'

    def filtering(self, tokens):
        # BxNxD
        class_token = tokens[:, 0:1]
        local_tokens = tokens[:, 1:]
        b, n, d = local_tokens.shape

        feature_maps = local_tokens.reshape(b, d, 16, 8)

        masked_feat, _ = self.mask_module(feature_maps)

        masked_tokens = masked_feat.reshape(b, n, d)
        tokens = torch.cat([class_token, masked_tokens*n], dim=1)
        return tokens

    def forward(self, x, clothes, label=None, cam_label=None, view_label=None, forzen=False):

        # BxNxD
        bio_feats = self.base(x)

        # bio_feats = self.filtering(bio_feats)
        #
        # bio_feats = self.weight_block(bio_feats)

        clot_feats = self.clothes(clothes)

        fusion_feat = torch.cat([bio_feats, clot_feats], dim=1)

        # Bx2NxD
        f = self.fusion(fusion_feat)

        bio = self.feat_bn(bio_feats[:, 0])
        f = self.fusion_feat_bn(torch.mean(f, dim=1))

        if not self.training:
            return None, None, F.normalize(f)

        logits = self.classifier(bio)
        f_logits = self.classifier_f(f)
        return bio, f, logits, f_logits


    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class PASS_Transformer_DualAttn_joint(nn.Module):
    def __init__(self, num_classes, net_config, num_cams=None, this_task_info=None):
        self.net_config = net_config
        super(PASS_Transformer_DualAttn_joint, self).__init__()
        if net_config.vit_type=='base':
            # self.base = vit_base_patch16_224_TransReID(img_size=(256,128), sie_xishu=3.0, local_feature=True, camera=0, view=0, stride_size=[12, 12], drop_path_rate=0.1)
            model_path = '/workspace/model_dir/code_ALBEF/pass_vit_base_full.pth'
            self.visual_encoder = vit_base_patch16_224_TransReID(img_size=(256,128), sie_xishu=3.0, camera=0, view=0, stride_size=[16, 16], drop_path_rate=0.1, drop_rate=0.0,attn_drop_rate=0.0, gem_pool=False, stem_conv=False)
            # self.visual_encoder = VisionTransformer(
            # img_size=256, patch_size=16, embed_dim=768, depth=12, num_heads=12,
            # mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), )
            self.visual_encoder_m = VisionTransformer(
            img_size=128, patch_size=16, embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), )
            # for k,v in self.visual_encoder_m.named_parameters():
            #     v.requires_grad=False
            # import pdb;pdb.set_trace()
            self.tokenizer = BertTokenizer.from_pretrained('/workspace/model_dir/code_ALBEF/instructReID/bert-base-uncased')
            
            bert_config = BertConfig.from_json_file('/workspace/model_dir/code_ALBEF/instructReID/config_bert.json')
            self.text_encoder = BertForMaskedLM.from_pretrained('/workspace/model_dir/code_ALBEF/instructReID/bert-base-uncased', config=bert_config)
            
            self.text_encoder_m = BertForMaskedLM.from_pretrained('/workspace/model_dir/code_ALBEF/instructReID/bert-base-uncased', config=bert_config)
            # for k,v in self.text_encoder_m.named_parameters():
            #     v.requires_grad=False
            self.text_width = self.text_encoder.config.hidden_size
            
            # if model_path != '':
            #     self.visual_encoder.load_param(model_path,hw_ratio=2)
        else:
            self.base = deit_small_patch16_224_TransReID(patch_size=net_config.patch_size_bio, stride_size=net_config.stride_size_bio,)
        # attn_drop_rate = net_config.dropout_clo
        # patch_size = net_config.patch_size_clo
        # stride_size = net_config.stride_size_clo
        # if net_config.vit_type=='base':
        #     # self.py_model_clip, self.transform_clip = clip.load('/mnt/lustre/tangshixiang/hwz/CLIP/ViT-B-32.pt', device='cuda', jit=False)
        #     # self.py_model_clip = CLIPModel.from_pretrained('./fashion_clip_model')
        #     # for k,v in self.py_model_clip.named_parameters():
        #     #     v.requires_grad=False
        #     self.clothes = nn.Linear(512, 768, bias=False)
        #     self.project = nn.Linear(512, 129*768, bias=False)
        # else:
        #     self.clothes = deit_small_patch16_224_TransReID(img_size=(128, 128), patch_size=patch_size, stride_size=stride_size, attn_drop_rate=attn_drop_rate)
        if net_config.vit_type=='base':
            self.in_planes = 768
            self.num_features = 768
        else:
            self.in_planes = 384
        self.temp = nn.Parameter(torch.ones([]) * 0.07)
        self.num_classes = num_classes
        self.vision_proj = nn.Linear(768, 256)
        self.text_proj = nn.Linear(768, 256)
        self.text_proj_m = nn.Linear(768, 256)
        self.multi_neck = True
        self.feat_fusion = 'cat'
        self.neck_feat = 'before'
        self.momentum = 0.995
        self.model_pairs = [[self.text_encoder, self.text_encoder_m],
                            [self.text_proj, self.text_proj_m],
                            ]
        self.copy_params()
        self.project = nn.Linear(768, 768, bias=False)
        # if self.feat_fusion == 'cat':
        #     self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        #     self.classifier.apply(weights_init_classifier)
        
        # if self.feat_fusion == 'mean':
        #     self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        #     self.classifier.apply(weights_init_classifier)
        
        # if self.multi_neck:
        #     self.bottleneck = nn.BatchNorm1d(self.in_planes)
        #     self.bottleneck.bias.requires_grad_(False)
        #     self.bottleneck.apply(weights_init_kaiming)
        # else:
        #     if self.feat_fusion == 'cat':
        #         self.bottleneck = nn.BatchNorm1d(self.in_planes*2)
        #         self.bottleneck.bias.requires_grad_(False)
        #         self.bottleneck.apply(weights_init_kaiming)
        #     elif self.feat_fusion == 'mean':
        #         self.bottleneck = nn.BatchNorm1d(self.in_planes)
        #         self.bottleneck.bias.requires_grad_(False)
        #         self.bottleneck.apply(weights_init_kaiming)

        # self.dropout = nn.Dropout(self.dropout_rate)
        
        # fusion_layers = []
        # for i in range(net_config.vit_fusion_layer):
        #     if net_config.attn_type=='fc':
        #         fusion_layers.append(torch.nn.Linear(self.num_features*2, self.num_features*2))
        #     else:
        #         fusion_layers.append(copy.deepcopy(self.base.blocks[-i]))
        # self.fusion = nn.Sequential(*fusion_layers)
            
        # self.norm = nn.LayerNorm(self.num_features, eps=1e-6)
        if self.net_config.fusion_loss=='all':
            self.feat_bn = nn.BatchNorm1d(self.num_features)
            self.feat_bn.bias.requires_grad_(False)
            init.constant_(self.feat_bn.weight, 1)
            init.constant_(self.feat_bn.bias, 0)

        self.fusion_feat_bn = nn.BatchNorm1d(self.num_features)
        self.fusion_feat_bn.bias.requires_grad_(False)
        init.constant_(self.fusion_feat_bn.weight, 1)
        init.constant_(self.fusion_feat_bn.bias, 0)
        # self.fusion_feat_bn_cat = nn.BatchNorm1d(2*self.num_features)
        # self.fusion_feat_bn_cat.bias.requires_grad_(False)
        # init.constant_(self.fusion_feat_bn_cat.weight, 1)
        # init.constant_(self.fusion_feat_bn_cat.bias, 0)
        # if 't2i' in this_task_info.task_name:
        #     if self.net_config.fusion_loss=='all':
        #         self.classifier = nn.Linear(self.num_features, 1, bias=False)
        #         init.normal_(self.classifier.weight, std=0.001)
        #     if 'bio' in net_config.fusion_branch:
        #         self.classifier_f = nn.Linear(self.num_features, 1, bias=False)
        #         init.normal_(self.classifier_f.weight, std=0.001)
        #     if 'clo' in net_config.fusion_branch:
        #         self.classifier_c = nn.Linear(self.num_features, 1, bias=False)
        #         init.normal_(self.classifier_c.weight, std=0.001)
        # else:
        # import pdb;pdb.set_trace()
        if self.net_config.fusion_loss=='all':
            self.classifier = nn.Linear(self.num_features, num_classes, bias=False)
            init.normal_(self.classifier.weight, std=0.001)
        if 'bio' in net_config.fusion_branch:
            self.classifier_f = nn.Linear(self.num_features, num_classes, bias=False)
            init.normal_(self.classifier_f.weight, std=0.001)
        if 'clo' in net_config.fusion_branch:
            self.classifier_c = nn.Linear(self.num_features, num_classes, bias=False)
            init.normal_(self.classifier_c.weight, std=0.001)
        
        # self.bottleneck = nn.BatchNorm1d(1024)
        # self.bottleneck.bias.requires_grad_(False)
        # self.bottleneck.apply(weights_init_kaiming)
        # self.dropout = nn.Dropout(0.0)
        self.itm_head = nn.Linear(self.text_width, 2)
        self.prd_head = nn.Linear(self.text_width, 2)
        self.mrtd_head = nn.Linear(self.text_width, 2)
        
        self.register_buffer("image_queue", torch.randn(256, 65536))
        self.register_buffer("text_queue", torch.randn(256, 65536))
        self.register_buffer("idx_queue", torch.full((1, 65536), -100))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)
        
        # self.fusion_feat_bn = nn.BatchNorm1d(self.num_features)
        # # self.fusion_feat_bn = nn.LayerNorm(self.num_features, eps=1e-6)
        # self.fusion_feat_bn.bias.requires_grad_(False)
        # init.constant_(self.fusion_feat_bn.weight, 1)
        # init.constant_(self.fusion_feat_bn.bias, 0)
        # if self.net_config.fusion_loss=='all':
        #     self.classifier = nn.Linear(self.num_features, num_classes, bias=False)
        #     init.normal_(self.classifier.weight, std=0.001)
        # if 'bio' in net_config.fusion_branch:
        #     self.classifier_f = nn.Linear(self.num_features, num_classes, bias=False)
        #     init.normal_(self.classifier_f.weight, std=0.001)
        # if 'clo' in net_config.fusion_branch:
        #     self.classifier_c = nn.Linear(self.num_features, num_classes, bias=False)
        #     init.normal_(self.classifier_c.weight, std=0.001)
        fusion_layers = []
        for i in range(net_config.vit_fusion_layer):
            if net_config.attn_type=='fc':
                fusion_layers.append(torch.nn.Linear(self.num_features*2, self.num_features*2))
            else:
                fusion_layers.append(copy.deepcopy(self.visual_encoder.blocks[-i]))
        self.fusion = nn.Sequential(*fusion_layers)
        # self.transformer_cross_attention_layers = nn.ModuleList()
        # for _ in range(1):
        #     self.transformer_cross_attention_layers.append(CrossAttentionLayer(d_model=768, nhead=12, dropout=0.0))
        
    def dual_attn(self, bio_feats, clot_feats, project_feats=None, project_feats_down=None):
        bio_class = bio_feats[:, 0:1]
        clot_class = clot_feats[:, 0:1]
        
        bio_fusion = torch.cat([bio_class, clot_feats[:, 1:]], dim=1)
        clot_fusion = torch.cat([clot_class, bio_feats[:, 1:]], dim=1)

        bio_fusion = self.fusion(bio_fusion)
        clot_fusion = self.fusion(clot_fusion)
        return bio_fusion, clot_fusion

    # def mix_attn(self, bio_feats, clo_feats):
    #     bio_clot_fusion = torch.cat([bio_feats, clo_feats], dim=1)
    #     output_fusion = self.fusion(bio_clot_fusion)
    #     return output_fusion
    
    # def fc_attn(self,bio_feats, clot_feats):
    #     num_batch = bio_feats.size(0)
    #     bio_class = bio_feats[:,0:1].reshape(num_batch,-1)
    #     clot_class = clot_feats[:,0:1].reshape(num_batch,-1)
    #     bio_clot_fusion = torch.cat([bio_class,clot_class],dim=1)
    #     bio_clot_fusion = self.fusion(bio_clot_fusion)
    #     bio_fusion = bio_clot_fusion[:, :self.num_features]
    #     clot_fusion = bio_clot_fusion[:, self.num_features:]
    #     return bio_fusion, clot_fusion

    def forward(self, x, instruction, this_task_info=None, label=None, cam_label=None, view_label=None, forzen=False):
        # BxNxD
        if self.training:
            if ('attr' in this_task_info.task_name or 'sc' in this_task_info.task_name or 't2i' in this_task_info.task_name or 'cross' in this_task_info.task_name) and 'ctcc' not in this_task_info.task_name:
                instruction_text = self.tokenizer(instruction, padding='max_length', max_length=70, return_tensors="pt").to('cuda')
                # extract text features
                text_output = self.text_encoder.bert(instruction_text.input_ids, attention_mask=instruction_text.attention_mask, return_dict=True, mode='text')
                text_embeds = text_output.last_hidden_state
                text_embeds_s = text_embeds[:,0]
                text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)
                text_attrs = instruction_text.attention_mask
                # project_feats = self.project(text_embeds_s.float()).unsqueeze(1)
            else :
                text_embeds = self.visual_encoder_m(instruction)
                text_embeds_s = text_embeds[:,0]
                text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)
                text_attrs = torch.ones(text_embeds.size()[:-1], dtype=torch.long).to(text_embeds.device)
                # project_feats = self.project(text_embeds_s.float()).unsqueeze(1)
                # clot_feats_s_ = self.clothes(clot_feats_s.float())
                # project_feats = self.project(clot_feats_s.float()).unsqueeze(1)
        else:
            if ('attr' in self.net_config.test_task_type or 'sc' in self.net_config.test_task_type or 't2i' in self.net_config.test_task_type or 'cross' in self.net_config.test_task_type) and 'ctcc' not in self.net_config.test_task_type:
                instruction_text = self.tokenizer(instruction, padding='max_length', max_length=70, return_tensors="pt").to('cuda')
                # extract text features
                text_output = self.text_encoder.bert(instruction_text.input_ids, attention_mask=instruction_text.attention_mask, return_dict=True, mode='text')
                text_embeds = text_output.last_hidden_state
                text_embeds_s = text_embeds[:,0]
                text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)
                text_attrs = instruction_text.attention_mask
                # project_feats = self.project(text_embeds_s.float()).unsqueeze(1)
            else:
                # clot_feats_s, clot_feats = self.py_model_clip.get_image_features(instruction)
                text_embeds = self.visual_encoder_m(instruction)
                text_embeds_s = text_embeds[:,0]
                text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)
                text_attrs = torch.ones(text_embeds.size()[:-1], dtype=torch.long).to(text_embeds.device)
                # project_feats = self.project(text_embeds_s.float()).unsqueeze(1)
        if this_task_info:
            task_name = this_task_info.task_name
        else:
            task_name = self.net_config.test_task_type
        
        # global_feat, local_feat_1, local_feat_2, local_feat_3, local_feat_all = self.visual_encoder(x)
        # image_embeds = torch.cat((global_feat.unsqueeze(1), local_feat_all), dim=1)
        # # image_embeds = self.visual_encoder(x)
        # image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(x.device)
        # image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
        
        # bio_fusion, clot_fusion = self.dual_attn(image_embeds, text_embeds)
        # feat = self.feat_bn(global_feat)
        # # import pdb;pdb.set_trace()
        # bio_f = self.fusion_feat_bn(bio_fusion[:, 0])
        # clot_f = self.fusion_feat_bn(clot_fusion[:, 0])
        
        # if not self.training:
        #     if 't2i' in self.net_config.test_task_type:
        #         return image_embeds, text_embeds, text_attrs, image_feat, text_feat, text_embeds
        #     else:
        #         return None, None, torch.cat([bio_f, clot_f], dim=1), text_embeds_s, text_embeds
        
        # idx = label.view(-1, 1)
        # idx_all = torch.cat([idx.t(), self.idx_queue.clone().detach()], dim=1)
        # # idx_all = torch.cat([idx.t()], dim=1)
        # pos_idx = torch.eq(idx, idx_all).float()
        # sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)
        # for line_idx in range(len(sim_targets)):
        #     sim_targets[line_idx][line_idx] = 1
        # with torch.no_grad():
        #     text_feat_all = torch.cat([text_feat.t(), self.text_queue.clone().detach()], dim=1)
        #     # text_feat_all = torch.cat([text_feat.t()], dim=1)
            
        #     image_feat_all = torch.cat([image_feat.t(), self.image_queue.clone().detach()], dim=1)
        #     # image_feat_all = torch.cat([image_feat.t()], dim=1)
        # sim_i2t = image_feat @ text_feat_all / self.temp
        # sim_t2i = text_feat @ image_feat_all / self.temp
        # sim_i2i = image_feat @ image_feat_all / self.temp
        # sim_t2t = text_feat @ text_feat_all / self.temp
        
        # loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_targets, dim=1).mean()
        # loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_targets, dim=1).mean()
        # loss_i2i = -torch.sum(F.log_softmax(sim_i2i, dim=1) * sim_targets, dim=1).mean()
        # loss_t2t = -torch.sum(F.log_softmax(sim_t2t, dim=1) * sim_targets, dim=1).mean()
        # loss_cl = (loss_i2t + loss_t2i + loss_i2i + loss_t2t) / 4
        
        # if 't2i' in task_name:
        #     self._dequeue_and_enqueue(image_feat, text_feat, idx)
        
        # output_pos = self.text_encoder.bert(encoder_embeds=text_embeds,
        #                                     attention_mask=text_attrs,
        #                                     encoder_hidden_states=image_embeds,
        #                                     encoder_attention_mask=image_atts,
        #                                     return_dict=True,
        #                                     mode='fusion',
        #                                     )
        
        # # label = label.view(-1, 1)
        # # label_all = torch.cat([label.t(), self.idx_queue.clone().detach()], dim=1)
        # # # label_all = torch.cat([label.t()], dim=1)
        # # pos_idx = torch.eq(label, label_all).float()
        
        # # with torch.no_grad():
        # #     bs = image_feat.size(0)
        # #     weights_i2t = F.softmax(sim_i2t[:, :bs], dim=1)
        # #     weights_t2i = F.softmax(sim_t2i[:, :bs], dim=1)
        # #     mask = torch.eq(label, label.T)
        # #     weights_i2t.masked_fill_(mask, 0)
        # #     weights_t2i.masked_fill_(mask, 0)
        # # neg_idx = torch.multinomial(weights_t2i, 1).flatten()
        # # image_embeds_n = image_embeds[neg_idx]
        
        # with torch.no_grad():
        #     bs = image_feat.size(0)
        #     weights_i2t = F.softmax(sim_i2t[:, :bs], dim=1)
        #     weights_t2i = F.softmax(sim_t2i[:, :bs], dim=1)
        #     mask = torch.eq(idx, idx.T)
        #     weights_i2t.masked_fill_(mask, 0)
        #     weights_t2i.masked_fill_(mask, 0)
        # neg_idx = torch.multinomial(weights_t2i, 1).flatten()
        # image_embeds_n = image_embeds[neg_idx]
        
        # text_neg_idx = torch.multinomial(weights_i2t, 1).flatten()
        # text_embeds_n = text_embeds[text_neg_idx]
        # text_attrs_n = text_attrs[text_neg_idx]
        
        
        # text_embeds_all = torch.cat([text_embeds, text_embeds_n], dim=0)
        # text_attrs_all = torch.cat([text_attrs, text_attrs_n], dim=0)
        # image_embeds_all = torch.cat([image_embeds_n, image_embeds], dim=0)
        # image_atts_all = torch.cat([image_atts, image_atts], dim=0)
        
        
        # output_neg_cross = self.text_encoder.bert(encoder_embeds=text_embeds_all,
        #                                         attention_mask=text_attrs_all,
        #                                         encoder_hidden_states=image_embeds_all,
        #                                         encoder_attention_mask=image_atts_all,
        #                                         return_dict=True,
        #                                         mode='fusion',
        #                                         )
        
        
        # vl_embeddings = torch.cat([output_pos.last_hidden_state[:, 0, :], output_neg_cross.last_hidden_state[:, 0, :]], dim=0)
        # vl_labels = torch.cat([torch.ones(output_pos.last_hidden_state.shape[0], dtype=torch.long), torch.zeros(output_neg_cross.last_hidden_state.shape[0], dtype=torch.long)], dim=0).to(vl_embeddings.device)
        # shuffle_idx = random.sample(range(vl_embeddings.shape[0]), vl_embeddings.shape[0])
        # # import pdb;pdb.set_trace()
        # vl_output = self.itm_head(vl_embeddings[shuffle_idx])
        # loss_pitm = F.cross_entropy(vl_output, vl_labels[shuffle_idx])
        
        # final_feat_after = feat
        
        # logits = self.classifier(final_feat_after)
        
        # f_logits = self.classifier_f(bio_f)
        # c_logits = self.classifier_c(clot_f)
        
        # if 't2i' in task_name:
        #     return image_feat, text_feat, output_pos, output_neg_cross, logits, vl_output, vl_labels[shuffle_idx], loss_cl, loss_pitm
        # else:
        #     return global_feat, bio_f, clot_f, logits, f_logits, c_logits, text_embeds_s
        
        if 't2i' in task_name:
            global_feat, local_feat_1, local_feat_2, local_feat_3, local_feat_all = self.visual_encoder(x)
            image_embeds = torch.cat((global_feat.unsqueeze(1), local_feat_all), dim=1)
            # image_embeds = self.visual_encoder(x)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(x.device)
            image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
            
            if not self.training:
                # return image_feat, clot_feats, clot_attrs, image_feat, clot_feats_s, clot_feats
                return image_embeds, text_embeds, text_attrs, image_feat, text_feat, text_embeds
            
            idx = label.view(-1, 1)
            idx_all = torch.cat([idx.t(), self.idx_queue.clone().detach()], dim=1)
            # idx_all = torch.cat([idx.t()], dim=1)
            pos_idx = torch.eq(idx, idx_all).float()
            sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)
            # for line_idx in range(len(sim_targets)):
            #     sim_targets[line_idx][line_idx] = 1
                
            
            # global_feat, local_feat_1, local_feat_2, local_feat_3, local_feat_all = self.base(x, cam_label=cam_label, view_label=view_label)
            
            # image_embeds = torch.cat((global_feat.unsqueeze(1),local_feat_all), dim=1)
            
            with torch.no_grad():
                self._momentum_update()
                text_output_m = self.text_encoder_m.bert(instruction_text.input_ids, attention_mask=instruction_text.attention_mask, return_dict=True, mode='text')
                text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:, 0, :]), dim=-1)
                text_feat_all = torch.cat([text_feat_m.t(), self.text_queue.clone().detach()], dim=1)
                # text_feat_all = torch.cat([text_feat.t(), self.text_queue.clone().detach()], dim=1)
                # text_feat_all = torch.cat([text_feat.t()], dim=1)
                
                image_feat_all = torch.cat([image_feat.t(), self.image_queue.clone().detach()], dim=1)
                # image_feat_all = torch.cat([image_feat.t()], dim=1)
                sim_i2t_m = image_feat @ text_feat_all / self.temp
                sim_t2i_m = text_feat_m @ image_feat_all / self.temp
                sim_i2i_m = image_feat @ image_feat_all / self.temp
                sim_t2t_m = text_feat_m @ text_feat_all / self.temp

                sim_i2t_targets = 0.4 * F.softmax(sim_i2t_m, dim=1) + (1 - 0.4) * sim_targets
                sim_t2i_targets = 0.4 * F.softmax(sim_t2i_m, dim=1) + (1 - 0.4) * sim_targets
                sim_i2i_targets = 0.4 * F.softmax(sim_i2i_m, dim=1) + (1 - 0.4) * sim_targets
                sim_t2t_targets = 0.4 * F.softmax(sim_t2t_m, dim=1) + (1 - 0.4) * sim_targets
            
            
            sim_i2t = image_feat @ text_feat_all / self.temp
            sim_t2i = text_feat @ image_feat_all / self.temp
            sim_i2i = image_feat @ image_feat_all / self.temp
            sim_t2t = text_feat @ text_feat_all / self.temp
            
            loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()
            loss_i2i = -torch.sum(F.log_softmax(sim_i2i, dim=1) * sim_i2i_targets, dim=1).mean()
            loss_t2t = -torch.sum(F.log_softmax(sim_t2t, dim=1) * sim_t2t_targets, dim=1).mean()
            loss_cl = (loss_i2t + loss_t2i + loss_i2i + loss_t2t) / 4
            
            self._dequeue_and_enqueue(image_feat, text_feat_m, idx)
            
            # instruction_text = self.tokenizer(instruction, padding='longest', max_length=50, return_tensors="pt").to('cuda')
            # # extract text features
            # text_output = self.text_encoder_m.bert(instruction_text.input_ids, attention_mask=instruction_text.attention_mask, return_dict=True, mode='text')
            # text_embeds = text_output.last_hidden_state
            # text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)
            
            # bio_fusion, clot_fusion = self.dual_attn(bio_feats, clot_feats)
            # import pdb;pdb.set_trace()
            output_pos = self.text_encoder.bert(encoder_embeds=text_embeds,
                                                attention_mask=instruction_text.attention_mask,
                                                encoder_hidden_states=image_embeds,
                                                encoder_attention_mask=image_atts,
                                                return_dict=True,
                                                mode='fusion',
                                                )
            
            with torch.no_grad():
                bs = image_feat.size(0)
                weights_i2t = F.softmax(sim_i2t[:, :bs], dim=1)
                weights_t2i = F.softmax(sim_t2i[:, :bs], dim=1)
                mask = torch.eq(idx, idx.T)
                weights_i2t.masked_fill_(mask, 0)
                weights_t2i.masked_fill_(mask, 0)
            neg_idx = torch.multinomial(weights_t2i, 1).flatten()
            image_embeds_n = image_embeds[neg_idx]
            
            text_neg_idx = torch.multinomial(weights_i2t, 1).flatten()
            text_embeds_n = text_embeds[text_neg_idx]
            text_attrs_n = instruction_text.attention_mask[text_neg_idx]
            
            text_embeds_all = torch.cat([text_embeds, text_embeds_n], dim=0)
            text_attrs_all = torch.cat([instruction_text.attention_mask, text_attrs_n], dim=0)
            image_embeds_all = torch.cat([image_embeds_n, image_embeds], dim=0)
            image_atts_all = torch.cat([image_atts, image_atts], dim=0)
            
            output_neg_cross = self.text_encoder.bert(encoder_embeds=text_embeds_all,
                                                    attention_mask=text_attrs_all,
                                                    encoder_hidden_states=image_embeds_all,
                                                    encoder_attention_mask=image_atts_all,
                                                    return_dict=True,
                                                    mode='fusion',
                                                    )
            
            vl_embeddings = torch.cat([output_pos.last_hidden_state[:, 0, :], output_neg_cross.last_hidden_state[:, 0, :]], dim=0)
            vl_labels = torch.cat([torch.ones(output_pos.last_hidden_state.shape[0], dtype=torch.long), torch.zeros(output_neg_cross.last_hidden_state.shape[0], dtype=torch.long)], dim=0).to(vl_embeddings.device)
            shuffle_idx = random.sample(range(vl_embeddings.shape[0]), vl_embeddings.shape[0])
            # import pdb;pdb.set_trace()
            vl_output = self.itm_head(vl_embeddings[shuffle_idx])
            loss_pitm = F.cross_entropy(vl_output, vl_labels[shuffle_idx])
            # prd_output = self.prd_head(output_pos.last_hidden_state[:, 0, :])
            # loss_prd = F.cross_entropy(prd_output, replace)
            
            # Sensitivity-aware Learning: Masked Language Modeling + Momentum-based Replaced Token Detection
            input_ids = instruction_text.input_ids.clone()
            labels = input_ids.clone()
            mrtd_input_ids = input_ids.clone()
            # Masked Language Modeling
            probability_matrix = torch.full(labels.shape, 0.15)
            input_ids, labels = self.mask(input_ids, self.text_encoder.config.vocab_size, targets=labels, probability_matrix=probability_matrix)
            with torch.no_grad():
                logits_m = self.text_encoder_m(input_ids,
                                            attention_mask=instruction_text.attention_mask,
                                            encoder_hidden_states=image_embeds,
                                            encoder_attention_mask=image_atts,
                                            return_dict=True,
                                            return_logits=True,
                                            )
                prediction = F.softmax(logits_m, dim=-1)
            mlm_output = self.text_encoder(input_ids,
                                        attention_mask=instruction_text.attention_mask,
                                        encoder_hidden_states=image_embeds,
                                        encoder_attention_mask=image_atts,
                                        return_dict=True,
                                        labels=labels,
                                        soft_labels=prediction,
                                        alpha=0.4
                                        )
            loss_mlm = mlm_output.loss
            # Momentum-based Replaced Token Detection
            with torch.no_grad():
                probability_matrix = torch.full(labels.shape, 0.3)
                mrtd_input_ids = self.mask(mrtd_input_ids, self.text_encoder.config.vocab_size, probability_matrix=probability_matrix)
                # momentum module is used as generator
                mrtd_logits_m = self.text_encoder_m(mrtd_input_ids,
                                                attention_mask=instruction_text.attention_mask,
                                                encoder_hidden_states=image_embeds,
                                                encoder_attention_mask=image_atts,
                                                return_dict=True,
                                                return_logits=True,
                                                )
                weights = F.softmax(mrtd_logits_m, dim=-1)
                mrtd_input_ids, mrtd_labels = self.mrtd_mask_modeling(mrtd_input_ids, instruction_text.input_ids, instruction_text.attention_mask, weights)
            output_mrtd = self.text_encoder.bert(mrtd_input_ids,
                                                attention_mask=instruction_text.attention_mask,
                                                encoder_hidden_states=image_embeds,
                                                encoder_attention_mask=image_atts,
                                                return_dict=True,
                                                )
            mrtd_output = self.mrtd_head(output_mrtd.last_hidden_state.view(-1, self.text_width))
            loss_mrtd = F.cross_entropy(mrtd_output, mrtd_labels.view(-1))
            
            
            return image_feat, text_feat, output_pos, output_neg_cross, vl_output, vl_labels[shuffle_idx], loss_cl, loss_pitm, loss_mlm, loss_mrtd
        else:
            global_feat, local_feat_1, local_feat_2, local_feat_3, local_feat_all = self.visual_encoder(x)
            # global_feat, local_feat_1, local_feat_2, local_feat_3, local_feat_all = self.visual_encoder(x, project_feats=project_feats.repeat(1,132,1))
            # # import pdb;pdb.set_trace()
            image_embeds = torch.cat((global_feat.unsqueeze(1), local_feat_all), dim=1)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(x.device)
            
            # output_pos = self.text_encoder_m.bert(encoder_embeds=bio_feats,
            #                                     attention_mask=image_atts,
            #                                     encoder_hidden_states=text_embeds,
            #                                     encoder_attention_mask=text_attrs,
            #                                     return_dict=True,
            #                                     mode='fusion',
            #                                     )
            
            # bio_fusion, clot_fusion = self.dual_attn(bio_feats, clot_feats)
            bio_fusion, clot_fusion = self.dual_attn(image_embeds, text_embeds)
            # bio_fusion = output_pos.last_hidden_state
            # clot_fusion = output_pos.last_hidden_state
            # bio_fusion, clot_fusion = self.dual_attn_se(bio_feats, clot_feats, project_feats=project_feats.repeat(1,129,1))
            
            feat = self.feat_bn(global_feat)
            # feat = self.bottleneck(global_feat)
            # local_feat_1_bn = self.bottleneck_1(local_feat_1)
            # local_feat_2_bn = self.bottleneck_2(local_feat_2)
            # local_feat_3_bn = self.bottleneck_3(local_feat_3)
            # import pdb;pdb.set_trace()
            bio_f = self.fusion_feat_bn(bio_fusion[:, 0])
            clot_f = self.fusion_feat_bn(clot_fusion[:, 0])
            
            if not self.training:
                # if 'sc' in self.net_config.test_task_type and 'ctcc' not in self.net_config.test_task_type:
                return None, None, torch.cat([bio_f, clot_f], dim=1), text_embeds_s, text_embeds
                # else:
                # return None, None, torch.cat([bio_fusion[:, 0], clot_fusion[:, 0]], dim=1), clot_feats_s, clot_feats
            
            # final_feat_after = torch.cat((feat, local_feat_1_bn / 3 + local_feat_2_bn / 3 + local_feat_3_bn / 3), dim=1)
            # final_feat_after = torch.cat((feat, local_feat_1_bn), dim=1)
            final_feat_after = feat
            
            logits = self.classifier(final_feat_after)
            
            f_logits = self.classifier_f(bio_f)
            c_logits = self.classifier_c(clot_f)
            # if ('sc' in this_task_info.task_name) and 'ctcc' not in this_task_info.task_name:
            return global_feat, bio_f, clot_f, logits, f_logits, c_logits, text_embeds_s
            
    def load_param(self, trained_path):
        param_dict = torch.load(trained_path, map_location='cpu')
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))
    
    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient
                
    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat, idx):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)
        idxs = concat_all_gather(idx)
        batch_size = image_feats.shape[0]
        # image_feats = image_feat
        # text_feats = text_feat
        # idxs = idx
        # batch_size = image_feats.shape[0]
        ptr = int(self.queue_ptr)
        # replace the keys at ptr (dequeue and enqueue)
        empty = self.image_queue.size(1) - ptr
        if batch_size <= empty:
            self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
            self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
            self.idx_queue[:, ptr:ptr + batch_size] = idxs.T
        else:
            self.image_queue[:, ptr:] = image_feats[:empty].T
            self.text_queue[:, ptr:] = text_feats[:empty].T
            self.idx_queue[:, ptr:] = idxs[:empty].T
            self.image_queue[:, :batch_size - empty] = image_feats[empty:].T
            self.text_queue[:, :batch_size - empty] = text_feats[empty:].T
            self.idx_queue[:, :batch_size - empty] = idxs[empty:].T
        ptr = (ptr + batch_size) % 65536  # move pointer
        self.queue_ptr[0] = ptr

    def mask(self, input_ids, vocab_size, targets=None, masked_indices=None, probability_matrix=None):
        if masked_indices is None:
            masked_indices = torch.bernoulli(probability_matrix).bool()
        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False
        if targets is not None:
            targets[~masked_indices] = -100  # We only compute loss on masked tokens
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id
        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(input_ids.device)
        input_ids[indices_random] = random_words[indices_random]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        if targets is not None:
            return input_ids, targets
        else:
            return input_ids

    def mrtd_mask_modeling(self, mrtd_input_ids, ori_input_ids, attention_mask, weights):
        bs = mrtd_input_ids.size(0)
        weights = weights.view(-1, weights.size(-1))
        pred = torch.multinomial(weights, 1).view(bs, -1)
        pred[:, 0] = self.tokenizer.cls_token_id
        # pad_token_id is 0
        mrtd_input_ids = pred * attention_mask
        mrtd_labels = (pred != ori_input_ids) * attention_mask
        mrtd_labels[mrtd_input_ids == self.tokenizer.pad_token_id] = -100
        mrtd_labels[mrtd_input_ids == self.tokenizer.cls_token_id] = -100
        return mrtd_input_ids, mrtd_labels
    
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output