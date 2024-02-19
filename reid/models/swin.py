from __future__ import absolute_import

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from typing import Optional
from torch import Tensor
import copy
# from reid.models.backbone.vit import deit_small_patch16_224_TransReID ,vit_base_patch16_224_TransReID, Block, vit_small_patch16_224_TransReID
from reid.models.backbone.swin_transformer import swin_base_patch4_window7_224, swin_small_patch4_window7_224, swin_tiny_patch4_window7_224
from reid.models.layers.metric import build_metric
from sklearn.cluster import KMeans
import numpy as np
import clip
from transformers import CLIPModel, CLIPProcessor

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

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


class Swin_Transformer_DualAttn_multi(nn.Module):
    def __init__(self, num_classes, net_config, num_cams=None, this_task_info=None):
        self.rearrange = True
        self.net_config = net_config
        super(Swin_Transformer_DualAttn_multi, self).__init__()
        
        if net_config.vit_type=='base':
            # self.base = vit_base_patch16_224_TransReID(img_size=(256,128), sie_xishu=3.0, local_feature=True, camera=15, view=0, stride_size=[12, 12], drop_path_rate=0.1)
            # self.base = swin_base_patch4_window7_224(img_size=(384,128), sie_xishu=3.0, local_feature=True, camera=num_cams, view=0, stride_size=[16, 16], drop_path_rate=0.1)
            model_path = '/mnt/lustre/tangshixiang/hwz/multitaskreid/reid/models/backbone/pretrain_weights/swin_base_market.pth'
            self.base = swin_base_patch4_window7_224(img_size=(384,128), drop_path_rate=0.1, drop_rate= 0.0,attn_drop_rate=0.0, pretrained=model_path, convert_weights=True, semantic_weight=0.2)
            
            if model_path != '':
                self.base.init_weights(model_path)
        else:
            self.base = deit_small_patch16_224_TransReID(patch_size=net_config.patch_size_bio, stride_size=net_config.stride_size_bio,)
            
        attn_drop_rate = net_config.dropout_clo
        patch_size = net_config.patch_size_clo
        stride_size = net_config.stride_size_clo
        if net_config.vit_type=='base':
            self.py_model_clip = CLIPModel.from_pretrained('./fashion_clip_model')
            for k,v in self.py_model_clip.named_parameters():
                v.requires_grad=False
            self.clothes = nn.Linear(512, 768, bias=False)
            # self.project = nn.Linear(512, 211*768, bias=False)
            self.project = nn.Linear(512, 768, bias=False)
        else:
            self.clothes = deit_small_patch16_224_TransReID(img_size=(128, 128), patch_size = patch_size, stride_size=stride_size, attn_drop_rate=attn_drop_rate)
        if net_config.vit_type=='base':
            self.num_features = 768
        else:
            self.num_features = 384

        block_s = self.base.stages[-1]
        # layer_norm = self.base.norm
        layer_norm = nn.BatchNorm1d(self.in_planes)
        self.b1 = nn.Sequential(
            copy.deepcopy(block_s),
            copy.deepcopy(layer_norm)
        )
        self.b2 = nn.Sequential(
            copy.deepcopy(block_s),
            copy.deepcopy(layer_norm)
        )
        
        fusion_layers = []
        for i in range(net_config.vit_fusion_layer):
            if net_config.attn_type=='fc':
                fusion_layers.append(torch.nn.Linear(self.num_features*2, self.num_features*2))
            else:
                fusion_layers.append(copy.deepcopy(self.base.stages[-i]))
        self.fusion = nn.Sequential(*fusion_layers)
        
        self.transformer_cross_attention_layers = nn.ModuleList()
        for _ in range(1):
            self.transformer_cross_attention_layers.append(CrossAttentionLayer(d_model=768, nhead=12, dropout=0.0))
        
        self.norm = nn.LayerNorm(self.num_features, eps=1e-6)
        if self.net_config.fusion_loss=='all':
            self.feat_bn = nn.BatchNorm1d(self.num_features)
            self.feat_bn.bias.requires_grad_(False)
            init.constant_(self.feat_bn.weight, 1)
            init.constant_(self.feat_bn.bias, 0)
            
            self.feat_bn_1 = nn.BatchNorm1d(self.num_features)
            self.feat_bn_1.bias.requires_grad_(False)
            init.constant_(self.feat_bn_1.weight, 1)
            init.constant_(self.feat_bn_1.bias, 0)
            
            self.feat_bn_2 = nn.BatchNorm1d(self.num_features)
            self.feat_bn_2.bias.requires_grad_(False)
            init.constant_(self.feat_bn_2.weight, 1)
            init.constant_(self.feat_bn_2.bias, 0)
            
            self.feat_bn_3 = nn.BatchNorm1d(self.num_features)
            self.feat_bn_3.bias.requires_grad_(False)
            init.constant_(self.feat_bn_3.weight, 1)
            init.constant_(self.feat_bn_3.bias, 0)
            
            self.feat_bn_4 = nn.BatchNorm1d(self.num_features)
            self.feat_bn_4.bias.requires_grad_(False)
            init.constant_(self.feat_bn_4.weight, 1)
            init.constant_(self.feat_bn_4.bias, 0)

        self.fusion_feat_bn = nn.BatchNorm1d(self.num_features)
        self.fusion_feat_bn.bias.requires_grad_(False)
        init.constant_(self.fusion_feat_bn.weight, 1)
        init.constant_(self.fusion_feat_bn.bias, 0)
        
        self.fusion_feat_bn_1 = nn.BatchNorm1d(self.num_features)
        self.fusion_feat_bn_1.bias.requires_grad_(False)
        init.constant_(self.fusion_feat_bn_1.weight, 1)
        init.constant_(self.fusion_feat_bn_1.bias, 0)
        
        self.fusion_feat_bn_2 = nn.BatchNorm1d(self.num_features)
        self.fusion_feat_bn_2.bias.requires_grad_(False)
        init.constant_(self.fusion_feat_bn_2.weight, 1)
        init.constant_(self.fusion_feat_bn_2.bias, 0)
        
        self.fusion_feat_bn_3 = nn.BatchNorm1d(self.num_features)
        self.fusion_feat_bn_3.bias.requires_grad_(False)
        init.constant_(self.fusion_feat_bn_3.weight, 1)
        init.constant_(self.fusion_feat_bn_3.bias, 0)
        
        self.fusion_feat_bn_4 = nn.BatchNorm1d(self.num_features)
        self.fusion_feat_bn_4.bias.requires_grad_(False)
        init.constant_(self.fusion_feat_bn_4.weight, 1)
        init.constant_(self.fusion_feat_bn_4.bias, 0)
        
        if self.net_config.fusion_loss=='all':
            self.classifier = nn.Linear(self.num_features, num_classes, bias=False)
            init.normal_(self.classifier.weight, std=0.001)
            self.classifier_1 = nn.Linear(self.num_features, num_classes, bias=False)
            init.normal_(self.classifier_1.weight, std=0.001)
            self.classifier_2 = nn.Linear(self.num_features, num_classes, bias=False)
            init.normal_(self.classifier_2.weight, std=0.001)
            self.classifier_3 = nn.Linear(self.num_features, num_classes, bias=False)
            init.normal_(self.classifier_3.weight, std=0.001)
            self.classifier_4 = nn.Linear(self.num_features, num_classes, bias=False)
            init.normal_(self.classifier_4.weight, std=0.001)
            
        if 'bio' in net_config.fusion_branch:
            self.classifier_f = nn.Linear(self.num_features, num_classes, bias=False)
            init.normal_(self.classifier_f.weight, std=0.001)
            self.classifier_f_1 = nn.Linear(self.num_features, num_classes, bias=False)
            init.normal_(self.classifier_f_1.weight, std=0.001)
            self.classifier_f_2 = nn.Linear(self.num_features, num_classes, bias=False)
            init.normal_(self.classifier_f_2.weight, std=0.001)
            self.classifier_f_3 = nn.Linear(self.num_features, num_classes, bias=False)
            init.normal_(self.classifier_f_3.weight, std=0.001)
            self.classifier_f_4 = nn.Linear(self.num_features, num_classes, bias=False)
            init.normal_(self.classifier_f_4.weight, std=0.001)
        if 'clo' in net_config.fusion_branch:
            self.classifier_c = nn.Linear(self.num_features, num_classes, bias=False)
            init.normal_(self.classifier_c.weight, std=0.001)
            self.classifier_c_1 = nn.Linear(self.num_features, num_classes, bias=False)
            init.normal_(self.classifier_c_1.weight, std=0.001)
            self.classifier_c_2 = nn.Linear(self.num_features, num_classes, bias=False)
            init.normal_(self.classifier_c_2.weight, std=0.001)
            self.classifier_c_3 = nn.Linear(self.num_features, num_classes, bias=False)
            init.normal_(self.classifier_c_3.weight, std=0.001)
            self.classifier_c_4 = nn.Linear(self.num_features, num_classes, bias=False)
            init.normal_(self.classifier_c_4.weight, std=0.001)
    

    # def dual_attn(self, bio_feats, clot_feats, project_feats=None, project_feats_down=None):
    #     bio_class = bio_feats[:, 0:1]
    #     clot_class = clot_feats[:, 0:1]
        
    #     bio_fusion = torch.cat([bio_class, clot_feats[:, 1:]], dim=1)
    #     clot_fusion = torch.cat([clot_class, bio_feats[:, 1:]], dim=1)

    #     bio_fusion = self.fusion(bio_fusion)
    #     clot_fusion = self.fusion(clot_fusion)
    #     return bio_fusion, clot_fusion
    
    def dual_attn(self, bio_feats, clot_feats, project_feats=None, project_feats_down=None):
        bio_class = bio_feats[:, 0:1]
        clot_class = clot_feats[:, 0:1]
        
        if project_feats is not None:
            clot_fusion = self.transformer_cross_attention_layers[0](project_feats.permute(1,0,2), bio_feats.permute(1,0,2)).permute(1,0,2)
        
        bio_fusion = torch.cat([bio_class, clot_feats[:, 1:]], dim=1)
        bio_fusion = self.fusion(bio_fusion)
        clot_fusion = self.fusion(clot_fusion)
        return bio_fusion, clot_fusion

    def mix_attn(self, bio_feats, clo_feats):
        bio_clot_fusion = torch.cat([bio_feats, clo_feats], dim=1)
        output_fusion = self.fusion(bio_clot_fusion)
        return output_fusion
    
    def fc_attn(self,bio_feats, clot_feats):
        num_batch = bio_feats.size(0)
        bio_class = bio_feats[:,0:1].reshape(num_batch,-1)
        clot_class = clot_feats[:,0:1].reshape(num_batch,-1)
        bio_clot_fusion = torch.cat([bio_class,clot_class],dim=1)
        bio_clot_fusion = self.fusion(bio_clot_fusion)
        bio_fusion = bio_clot_fusion[:, :self.num_features]
        clot_fusion = bio_clot_fusion[:, self.num_features:]
        return bio_fusion, clot_fusion

    def forward(self, x, instruction, this_task_info=None, label=None, cam_label=None, view_label=None, forzen=False):
        # BxNxD
        if self.training:
            if ('attr' in this_task_info.task_name or 'cc' in this_task_info.task_name or 'sc' in this_task_info.task_name) and 'ctcc' not in this_task_info.task_name:
                clot_feats_s, clot_feats = self.py_model_clip.get_text_features(instruction.squeeze(dim=1))
                clot_feats = self.clothes(clot_feats.float())
            else:
                clot_feats_s, clot_feats = self.py_model_clip.get_image_features(instruction)
                clot_feats = self.clothes(clot_feats.float())
        else:
            if self.net_config.test_task_type == 'sc' or self.net_config.test_task_type == 'attr':
                clot_feats_s, clot_feats = self.py_model_clip.get_text_features(instruction.squeeze(dim=1))
            elif self.net_config.test_task_type == 'cc' or self.net_config.test_task_type == 'ctcc':
                clot_feats_s, clot_feats = self.py_model_clip.get_image_features(instruction)
            clot_feats = self.clothes(clot_feats.float())
        # project_feats = self.project(clot_feats_s.float()).view(x.shape[0],211,-1)
        project_feats = self.project(clot_feats_s.float()).view(x.shape[0],1,-1).repeat(1,211,1)
        # bio_feats = self.base(x, cam_label=cam_label*0, view_label=view_label)
        if not self.training:
            if self.net_config.test_task_type != 'sc':
                bio_feats = self.base(x, cam_label=cam_label*0, view_label=view_label)
            else:
                bio_feats = self.base(x, cam_label=cam_label, view_label=view_label)
        else:
            bio_feats = self.base(x, cam_label=cam_label, view_label=view_label)
        import pdb;pdb.set_trace()
        # global branch
        b1_feat = self.b1(bio_feats) # [64, 129, 768]
        global_feat = b1_feat[:, 0]
        
        # JPM branch
        feature_length = bio_feats.size(1) - 1
        patch_length = feature_length // 4
        token = bio_feats[:, 0:1]

        if self.rearrange:
            x = shuffle_unit(bio_feats, 5, 2)
        else:
            x = bio_feats[:, 1:]
        # lf_1
        b1_local_feat = x[:, :patch_length]
        b1_local_feat = self.b2(torch.cat((token, b1_local_feat), dim=1))
        local_feat_1 = b1_local_feat[:, 0]

        # lf_2
        b2_local_feat = x[:, patch_length:patch_length*2]
        b2_local_feat = self.b2(torch.cat((token, b2_local_feat), dim=1))
        local_feat_2 = b2_local_feat[:, 0]

        # lf_3
        b3_local_feat = x[:, patch_length*2:patch_length*3]
        b3_local_feat = self.b2(torch.cat((token, b3_local_feat), dim=1))
        local_feat_3 = b3_local_feat[:, 0]

        # lf_4
        b4_local_feat = x[:, patch_length*3:patch_length*4]
        b4_local_feat = self.b2(torch.cat((token, b4_local_feat), dim=1))
        local_feat_4 = b4_local_feat[:, 0]
        
        
        if self.net_config.attn_type=='dual_attn':
            bio_fusion, clot_fusion = self.dual_attn(b1_feat, clot_feats, project_feats=project_feats)
            bio_fusion_1, clot_fusion_1 = self.dual_attn(b1_local_feat, clot_feats, project_feats=project_feats)
            bio_fusion_2, clot_fusion_2 = self.dual_attn(b2_local_feat, clot_feats, project_feats=project_feats)
            bio_fusion_3, clot_fusion_3 = self.dual_attn(b3_local_feat, clot_feats, project_feats=project_feats)
            bio_fusion_4, clot_fusion_4 = self.dual_attn(b4_local_feat, clot_feats, project_feats=project_feats)
        elif self.net_config.attn_type=='mix':
            bio_fusion = self.mix_attn(bio_feats, clot_feats)
            clot_fusion = bio_fusion
        else:
            bio_fusion, clot_fusion = self.fc_attn(bio_feats, clot_feats)
        
        if self.net_config.fusion_loss=='all':
            bio = self.feat_bn(global_feat)
            bio_1 = self.feat_bn_1(local_feat_1)
            bio_2 = self.feat_bn_2(local_feat_2)
            bio_3 = self.feat_bn_3(local_feat_3)
            bio_4 = self.feat_bn_4(local_feat_4)
            
        else:
            bio = bio_feats[:,0]
        if self.net_config.attn_type=='fc':
            bio_f = self.fusion_feat_bn(bio_fusion)
            clot_f = self.fusion_feat_bn(clot_fusion)
        else:
            bio_f = self.fusion_feat_bn(bio_fusion[:, 0])
            clot_f = self.fusion_feat_bn(clot_fusion[:, 0])
            bio_f_1 = self.fusion_feat_bn_1(bio_fusion_1[:, 0])
            clot_f_1 = self.fusion_feat_bn_1(clot_fusion_1[:, 0])
            bio_f_2 = self.fusion_feat_bn_2(bio_fusion_2[:, 0])
            clot_f_2 = self.fusion_feat_bn_2(clot_fusion_2[:, 0])
            bio_f_3 = self.fusion_feat_bn_3(bio_fusion_3[:, 0])
            clot_f_3 = self.fusion_feat_bn_3(clot_fusion_3[:, 0])
            bio_f_4 = self.fusion_feat_bn_4(bio_fusion_4[:, 0])
            clot_f_4 = self.fusion_feat_bn_4(clot_fusion_4[:, 0])

        
        bio_p = torch.cat([bio, bio_1 / 4, bio_2 / 4, bio_3 / 4, bio_4 / 4], dim=1)
        bio_fusion_p = torch.cat([bio_f, bio_f_1 / 4, bio_f_2 / 4, bio_f_3 / 4, bio_f_4 / 4], dim=1)
        clot_fusion_p = torch.cat([clot_f, clot_f_1 / 4, clot_f_2 / 4, clot_f_3 / 4, clot_f_4 / 4], dim=1)
        
        if not self.training:
            if self.net_config.test_task_type == 'attr' or self.net_config.test_task_type == 'ctcc':
                return None, None, F.normalize(torch.cat([bio_fusion_p, clot_fusion_p], dim=1)), clot_feats_s, clot_feats
            elif self.net_config.test_task_type == 'cc' or self.net_config.test_task_type == 'sc':
                return None, None, F.normalize(torch.cat([bio_p, bio_fusion_p, clot_fusion_p], dim=1)), clot_feats_s, clot_feats
        
        if self.net_config.fusion_loss=='all':
            logits = self.classifier(bio)
            logits_1 = self.classifier_1(bio_1)
            logits_2 = self.classifier_2(bio_2)
            logits_3 = self.classifier_3(bio_3)
            logits_4 = self.classifier_4(bio_4)
            logits_list = [logits, logits_1, logits_2, logits_3, logits_4]
        else:
            logits = None
        
        if 'bio' in self.net_config.fusion_branch:
            f_logits = self.classifier_f(bio_f)
            f_logits_1 = self.classifier_f_1(bio_f_1)
            f_logits_2 = self.classifier_f_2(bio_f_2)
            f_logits_3 = self.classifier_f_3(bio_f_3)
            f_logits_4 = self.classifier_f_4(bio_f_4)
            f_logits_list = [f_logits, f_logits_1, f_logits_2, f_logits_3, f_logits_4]
        else:
            f_logits = None
        
        if 'clo' in self.net_config.fusion_branch:
            c_logits = self.classifier_c(clot_f)
            c_logits_1 = self.classifier_c_1(clot_f_1)
            c_logits_2 = self.classifier_c_2(clot_f_2)
            c_logits_3 = self.classifier_c_3(clot_f_3)
            c_logits_4 = self.classifier_c_4(clot_f_4)
            c_logits_list = [c_logits, c_logits_1, c_logits_2, c_logits_3, c_logits_4]
        else:
            c_logits = None
        
        bio_list = [global_feat, local_feat_1, local_feat_2, local_feat_3, local_feat_4]
        bio_fusion_list = [bio_fusion[:, 0], bio_fusion_1[:, 0], bio_fusion_2[:, 0], bio_fusion_3[:, 0], bio_fusion_4[:, 0]]
        clot_fusion_list = [clot_fusion[:, 0], clot_fusion_1[:, 0], clot_fusion_2[:, 0], clot_fusion_3[:, 0], clot_fusion_4[:, 0]]
        
        return bio_list, bio_fusion_list, clot_fusion_list, logits_list, f_logits_list, c_logits_list, clot_feats_s


class Swin_Transformer_DualAttn(nn.Module):
    def __init__(self, num_classes, net_config, num_cams=None, this_task_info=None):
        self.net_config = net_config
        super(Swin_Transformer_DualAttn, self).__init__()
        if net_config.vit_type=='base':
            # self.base = vit_base_patch16_224_TransReID(img_size=(256,128), sie_xishu=3.0, local_feature=True, camera=0, view=0, stride_size=[12, 12], drop_path_rate=0.1)
            # model_path = '/mnt/lustre/tangshixiang/hwz/multitaskreid/reid/models/backbone/pretrain_weights/swin_base_market.pth'
            model_path = '/mnt/lustre/tangshixiang/hwz/multitaskreid/reid/models/backbone/pretrain_weights/swin_base.pth'
            self.base = swin_base_patch4_window7_224(img_size=(384,128), drop_path_rate=0.1, drop_rate= 0.0,attn_drop_rate=0.0, pretrained=model_path, convert_weights=True, semantic_weight=0.2)
            if model_path != '':
                self.base.init_weights(model_path)
        else:
            self.base = deit_small_patch16_224_TransReID(patch_size=net_config.patch_size_bio, stride_size=net_config.stride_size_bio,)
        attn_drop_rate = net_config.dropout_clo
        patch_size = net_config.patch_size_clo
        stride_size = net_config.stride_size_clo
        if net_config.vit_type=='base':
            self.py_model_clip = CLIPModel.from_pretrained('./fashion_clip_model')
            for k,v in self.py_model_clip.named_parameters():
                v.requires_grad=False
            self.clothes = nn.Linear(512, 768, bias=False)
            self.project = nn.Linear(512, 129*768, bias=False)
        else:
            self.clothes = deit_small_patch16_224_TransReID(img_size=(128, 128), patch_size = patch_size, stride_size=stride_size, attn_drop_rate=attn_drop_rate)
        if net_config.vit_type=='base':
            self.num_features = 1024
        else:
            self.num_features = 384
   
        fusion_layers = []
        for i in range(net_config.vit_fusion_layer):
            if net_config.attn_type=='fc':
                fusion_layers.append(torch.nn.Linear(self.num_features*2, self.num_features*2))
            else:
                fusion_layers.append(copy.deepcopy(self.base.stages[-i]))
        self.fusion = nn.Sequential(*fusion_layers)
            
        self.norm = nn.LayerNorm(self.num_features, eps=1e-6)
        if self.net_config.fusion_loss=='all':
            self.feat_bn = nn.BatchNorm1d(self.num_features)
            self.feat_bn.bias.requires_grad_(False)
            init.constant_(self.feat_bn.weight, 1)
            init.constant_(self.feat_bn.bias, 0)

        self.fusion_feat_bn = nn.BatchNorm1d(self.num_features)
        self.fusion_feat_bn.bias.requires_grad_(False)
        init.constant_(self.fusion_feat_bn.weight, 1)
        init.constant_(self.fusion_feat_bn.bias, 0)
        if self.net_config.fusion_loss=='all':
            self.classifier = nn.Linear(self.num_features, num_classes, bias=False)
            init.normal_(self.classifier.weight, std=0.001)
        if 'bio' in net_config.fusion_branch:
            self.classifier_f = nn.Linear(self.num_features, num_classes, bias=False)
            init.normal_(self.classifier_f.weight, std=0.001)
        if 'clo' in net_config.fusion_branch:
            self.classifier_c = nn.Linear(self.num_features, num_classes, bias=False)
            init.normal_(self.classifier_c.weight, std=0.001)
        
        self.bottleneck = nn.BatchNorm1d(1024)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.dropout = nn.Dropout(0.0)
        
    def dual_attn(self, bio_feats, clot_feats, project_feats=None, project_feats_down=None):
        bio_class = bio_feats[:, 0:1]
        clot_class = clot_feats[:, 0:1]
        
        bio_fusion = torch.cat([bio_class, clot_feats[:, 1:]], dim=1)
        clot_fusion = torch.cat([clot_class, bio_feats[:, 1:]], dim=1)

        bio_fusion = self.fusion(bio_fusion)
        clot_fusion = self.fusion(clot_fusion)
        return bio_fusion, clot_fusion

    def mix_attn(self, bio_feats, clo_feats):
        bio_clot_fusion = torch.cat([bio_feats, clo_feats], dim=1)
        output_fusion = self.fusion(bio_clot_fusion)
        return output_fusion
    
    def fc_attn(self,bio_feats, clot_feats):
        num_batch = bio_feats.size(0)
        bio_class = bio_feats[:,0:1].reshape(num_batch,-1)
        clot_class = clot_feats[:,0:1].reshape(num_batch,-1)
        bio_clot_fusion = torch.cat([bio_class,clot_class],dim=1)
        bio_clot_fusion = self.fusion(bio_clot_fusion)
        bio_fusion = bio_clot_fusion[:, :self.num_features]
        clot_fusion = bio_clot_fusion[:, self.num_features:]
        return bio_fusion, clot_fusion

    def forward(self, x, instruction, this_task_info=None, label=None, cam_label=None, view_label=None, forzen=False):
        # BxNxD
        global_feat, featmaps = self.base(x)
        feat = self.bottleneck(global_feat)
        feat_cls = self.dropout(feat)
        if self.training:
            cls_score = self.classifier(feat_cls)
            return global_feat, global_feat, global_feat, cls_score, cls_score, cls_score, cls_score
        
        else:
            # print("Test with feature before BN")
            return None, None, global_feat, global_feat, global_feat
        # None, None, torch.cat([bio_fusion[:, 0], clot_fusion[:, 0]], dim=1), clot_feats_s, clot_feats
        
        
        
        if self.training:
            if ('attr' in this_task_info.task_name or 'cc' in this_task_info.task_name or 'sc' in this_task_info.task_name) and 'ctcc' not in this_task_info.task_name:
                clot_feats_s, clot_feats = self.py_model_clip.get_text_features(instruction.squeeze(dim=1))
                clot_feats = self.clothes(clot_feats.float())
            else:
                clot_feats_s, clot_feats = self.py_model_clip.get_image_features(instruction)
                clot_feats = self.clothes(clot_feats.float())
        else:
            if ('attr' in self.net_config.test_task_type or 'cc' in self.net_config.test_task_type or 'sc' in self.net_config.test_task_type) and 'ctcc' not in self.net_config.test_task_type:
                clot_feats_s, clot_feats = self.py_model_clip.get_text_features(instruction.squeeze(dim=1))
            else:
                clot_feats_s, clot_feats = self.py_model_clip.get_image_features(instruction)
            clot_feats = self.clothes(clot_feats.float())
            
        
        
        # project_feats = self.project(clot_feats_s.float()).view(x.shape[0],129,-1)
        bio_feats = self.base(x)
        bio_feats, featmaps = self.base(x)
        # self.bottleneck(global_feat)
        import pdb;pdb.set_trace()
        if self.net_config.attn_type=='dual_attn':
            bio_fusion, clot_fusion = self.dual_attn(bio_feats, clot_feats)
        elif self.net_config.attn_type=='mix':
            bio_fusion = self.mix_attn(bio_feats, clot_feats)
            clot_fusion = bio_fusion
        else:
            bio_fusion, clot_fusion = self.fc_attn(bio_feats, clot_feats)
        if self.net_config.fusion_loss=='all':
            bio = self.feat_bn(bio_feats[:, 0])
            
        else:
            bio = bio_feats[:,0]
        if self.net_config.attn_type=='fc':
            bio_f = self.fusion_feat_bn(bio_fusion)
            clot_f = self.fusion_feat_bn(clot_fusion)
        else:
            bio_f = self.fusion_feat_bn(bio_fusion[:, 0])
            clot_f = self.fusion_feat_bn(clot_fusion[:, 0])

        if not self.training:
            return None, None, torch.cat([bio_fusion[:, 0], clot_fusion[:, 0]], dim=1), clot_feats_s, clot_feats
            
        if self.net_config.fusion_loss=='all':
            logits = self.classifier(bio)
        else:
            logits = None
        if 'bio' in self.net_config.fusion_branch:
            f_logits = self.classifier_f(bio_f)
        else:
            f_logits = None
        if 'clo' in self.net_config.fusion_branch:
            c_logits = self.classifier_c(clot_f)
        else:
            c_logits = None

        return bio_feats[:, 0], bio_fusion[:, 0], clot_fusion[:, 0], logits, f_logits, c_logits, clot_feats_s