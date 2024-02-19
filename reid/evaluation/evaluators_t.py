from __future__ import print_function, absolute_import

import time
from collections import OrderedDict

import numpy as np
import torch
import os
from reid.utils import to_torch, to_numpy
from reid.utils.meters import AverageMeter
import random
import shutil
import cv2
from reid.utils.vit_rollout import show_mask_on_image
import json
class Evaluator(object):
    def __init__(self, model, validate_feat):
        super(Evaluator, self).__init__()
        self.model = model
        self.validate_feat = validate_feat
        json_list = ' <your project root> + instructReID/data/cuhk_pedes/caption_t2i_v2.json'
        attr_file =  open(json_list, 'r', encoding='utf-8')
        self.attr_dict = json.load(attr_file)

    def visualize(self, vis_feat, data_loader ,attention_rollout, root, result_root, print_freq=50):
        self.model.eval()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        features = OrderedDict()
        labels = OrderedDict()
        end = time.time()

        
        for i, data in enumerate(data_loader):
            imgs=data[0]
            clothes=data[1]
            if vis_feat=='b':
                #imgs = data[0]
                fnames = data[2]
            else:
                #imgs = data[1]
                fnames = data[3]
            pids = data[4]

            data_time.update(time.time() - end)

            input_tensor = to_torch(imgs)
            clothes_tensor = to_torch(clothes)
            print('input_tensor shape:',input_tensor.shape)
            print('clothes_tensor shape:',clothes_tensor.shape)
            rgb_img = cv2.imread(os.path.join(root, fnames[0]))[:, :, ::-1]
            if vis_feat=='b':
                rgb_img = cv2.resize(rgb_img, input_tensor.shape[-2:][::-1])
            else:
                rgb_img = cv2.resize(rgb_img, clothes_tensor.shape[-2:][::-1])
            rgb_img = np.float32(rgb_img)
            grayscale_cam = attention_rollout(input_tensor=input_tensor, clothes_tensor=clothes_tensor)
            # grayscale_cam = grayscale_cam[0, :]
            mask = cv2.resize(grayscale_cam, (rgb_img.shape[1], rgb_img.shape[0]))
            mask = show_mask_on_image(rgb_img, mask)
            
            cv2.imwrite(os.path.join(result_root, '-'.join(fnames[0].split('/'))), mask)

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Extract Features: [{}/{}]\t'
                        'Time {:.3f} ({:.3f})\t'
                        'Data {:.3f} ({:.3f})\t'
                        .format(i + 1, len(data_loader),
                                batch_time.val, batch_time.avg,
                                data_time.val, data_time.avg))
    @torch.no_grad()
    def vis_retrieval(self, data_loader, query, gallery, source_root, save_root):
        features, _, features_bio, features_clot, attr_clot = self.extract_features(self.model, data_loader, self.validate_feat)
        # distmat = self.pairwise_distance_t2i(features, query, gallery)
        distmat, x_f, y_f, x_f_attr = self.pairwise_distance_t2i_f(features, features_bio, features_clot, attr_clot, query, gallery)
        
        score_matrix_t2i = torch.full((len(x_f), len(y_f)), -100.0).to('cuda')
        # import pdb;pdb.set_trace()
        
        for i, sims in enumerate(distmat):
            topk_distmat, topk_idx = sims.topk(k=128, dim=0)
            encoder_output = y_f[topk_idx]
            text_feats = x_f[i].repeat(128, 1, 1)
            text_atts = x_f_attr[i].repeat(128, 1)
            # import pdb;pdb.set_trace()
            
            encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to('cuda')
            output = self.model.module.text_encoder.bert(encoder_embeds=text_feats.to('cuda'), attention_mask=text_atts.to('cuda'), encoder_hidden_states=encoder_output.to('cuda'), encoder_attention_mask=encoder_att, return_dict=True, mode='fusion')
            # bio_fusion, clot_fusion = self.model.module.dual_attn(img_feats_topk.to('cuda').detach(), text_feats.to('cuda').detach())
            # vl_embeddings = torch.cat([bio_fusion[:, 0], clot_fusion[:, 0]], dim=-1)
            # # vl_f = self.model.module.fusion_feat_bn_cat(vl_embeddings.detach())
            # vl_f = vl_embeddings
            score = self.model.module.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
            score_matrix_t2i[i, topk_idx] = score
        
        indices = torch.argsort(score_matrix_t2i, dim=-1, descending=True)
        
        # distmat = self.pairwise_distance(features, query, gallery)
        distmat = to_numpy(distmat)
        query_ids = [items[2] for items in query]
        gallery_ids = [items[2] for items in gallery]
        query_imgs = [items[0] for items in query]
        gallery_imgs = [items[0] for items in gallery]
        query_clo_imgs = [items[1] for items in query]
        # indices = np.argsort(distmat, axis=1)
        num_query = len(indices)
        query_idxs = list(range(num_query))
        #random.shuffle(query_idxs)
        for query_idx in query_idxs:
            indice = indices[query_idx]
            save_root_ = os.path.join(save_root, str(query_idx))
            if not os.path.exists(save_root_):
                os.makedirs(save_root_)
            # import pdb;pdb.set_trace()
            with open(os.path.join(save_root_, 'query-'+str(query_idx)+'_id_'+str(query_ids[query_idx])+'.txt'), 'w')as f:
                f.write(self.attr_dict[query[query_idx][1]])
            # source_query_path = os.path.join(source_root, query_imgs[query_idx])
            # target_query_path = os.path.join(save_root_, 'query-'+str(query_idx)+'_id_'+str(query_ids[query_idx])+'.jpg')
            # shutil.copyfile(source_query_path, target_query_path)
            # source_clo_path = os.path.join(source_root, query_clo_imgs[query_idx])
            # target_clo_path = os.path.join(save_root_, 'clo.jpg')
            # shutil.copyfile(source_clo_path , target_clo_path)
            topk=1
            for gallery_idx in indice[:10]:
                source_gallery_path = os.path.join(source_root, gallery_imgs[gallery_idx])
                target_gallery_path = os.path.join(save_root_, 'top-')+str(topk)+'_id_'+str(gallery_ids[gallery_idx])+'.jpg'
                topk+=1
                shutil.copyfile(source_gallery_path,target_gallery_path)
    
    @torch.no_grad()
    def evaluate(self, data_loader, query, gallery):
        # features, labels, features_bio, features_clot
        features, _, features_bio, features_clot, attr_clot = self.extract_features(self.model, data_loader, self.validate_feat)
        # distmat = self.pairwise_distance_t2i(features, query, gallery)
        distmat, x_f, y_f, x_f_attr = self.pairwise_distance_t2i_f(features, features_bio, features_clot, attr_clot, query, gallery)
        
        score_matrix_t2i = torch.full((len(x_f), len(y_f)), -100.0).to('cuda')
        # import pdb;pdb.set_trace()
        
        for i, sims in enumerate(distmat):
            topk_distmat, topk_idx = sims.topk(k=128, dim=0)
            encoder_output = y_f[topk_idx]
            text_feats = x_f[i].repeat(128, 1, 1)
            text_atts = x_f_attr[i].repeat(128, 1)
            # import pdb;pdb.set_trace()
            
            encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to('cuda')
            output = self.model.module.text_encoder.bert(encoder_embeds=text_feats.to('cuda'), attention_mask=text_atts.to('cuda'), encoder_hidden_states=encoder_output.to('cuda'), encoder_attention_mask=encoder_att, return_dict=True, mode='fusion')
            # bio_fusion, clot_fusion = self.model.module.dual_attn(img_feats_topk.to('cuda').detach(), text_feats.to('cuda').detach())
            # vl_embeddings = torch.cat([bio_fusion[:, 0], clot_fusion[:, 0]], dim=-1)
            # # vl_f = self.model.module.fusion_feat_bn_cat(vl_embeddings.detach())
            # vl_f = vl_embeddings
            score = self.model.module.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
            score_matrix_t2i[i, topk_idx] = score
        
        return self.eval_f(distmat, query, gallery, score_matrix_t2i.cpu())
    
    @torch.no_grad()
    def eval_f(self, distmat, query, gallery, score_matrix_t2i):
        index = torch.argsort(score_matrix_t2i, dim=-1, descending=True)
        
        query_ids = [items[2] for items in query]
        gallery_ids = [items[2] for items in gallery]
        query_cams = [items[-1] for items in query]
        gallery_cams = [items[-1] for items in gallery]

        query_ids = torch.tensor(np.asarray(query_ids)).to('cuda')
        gallery_ids = torch.tensor(np.asarray(gallery_ids)).to('cuda')
        query_cams = np.asarray(query_cams)
        gallery_cams = np.asarray(gallery_cams)
        # import pdb;pdb.set_trace()
        pred_person = gallery_ids[index]
        matches = (query_ids.view(-1, 1).eq(pred_person)).long().cpu()

        def acc_k(matches, k=1):
            matches_k = matches[:, :k].sum(dim=-1)
            matches_k = torch.sum((matches_k > 0))
            return matches_k / matches.size(0)

        # Compute metrics
        ir1 = acc_k(matches, k=1).item()
        ir5 = acc_k(matches, k=5).item()
        ir10 = acc_k(matches, k=10).item()
        ir50 = acc_k(matches, k=50).item()
        ir_mean = (ir1 + ir5 + ir10) / 3

        real_num = matches.sum(dim=-1)
        tmp_cmc = matches.cumsum(dim=-1).float()
        order = torch.arange(start=1, end=matches.size(1) + 1, dtype=torch.long)
        tmp_cmc = tmp_cmc / order
        tmp_cmc *= matches
        AP = tmp_cmc.sum(dim=-1) / real_num
        mAP = AP.mean()
        eval_result = {'r1': ir1,
                    'r5': ir5,
                    'r10': ir10,
                    'r50': ir50,
                    'r_mean': ir_mean,
                    'mAP': mAP.item()
                    }
        

        print("=" * 80)
        # print('Mean AP: {:4.}'.format(mAP.item()))
        print('CMC Scores: ')
        print(eval_result)
        return mAP.item()
    
    def eval(self, distmat, query, gallery):
        distmat = to_numpy(distmat)

        query_ids = [items[2] for items in query]
        gallery_ids = [items[2] for items in gallery]
        query_cams = [items[-1] for items in query]
        gallery_cams = [items[-1] for items in gallery]

        query_ids = np.asarray(query_ids)
        gallery_ids = np.asarray(gallery_ids)
        query_cams = np.asarray(query_cams)
        gallery_cams = np.asarray(gallery_cams)

        cmc_scores, mAP = self.eval_func(distmat, q_pids=query_ids, g_pids=gallery_ids,
                                    q_camids=query_cams, g_camids=gallery_cams, max_rank=50)

        print("=" * 80)
        print('Mean AP: {:4.1%}'.format(mAP))
        print('CMC Scores:')
        cmc_topk = (1, 5, 10, 20, 50)
        for k in cmc_topk:
            print('  top-{:<4}{:12.1%}'.format(k, cmc_scores[k - 1]))
        return mAP

    @staticmethod
    def pairwise_distance(features, query=None, gallery=None):
        if query is None and gallery is None:
            n = len(features)
            x = torch.cat(list(features.values()))
            x = x.view(n, -1)
            dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
            dist_m = dist_m.expand(n, n) - 2 * torch.mm(x, x.t())
            return dist_m

        x = torch.cat([features[(f[0],f[1])].unsqueeze(0) for f in query], 0)
        y = torch.cat([features[(f[0],f[1])].unsqueeze(0) for f in gallery], 0)
        m, n = x.size(0), y.size(0)
        x = x.view(m, -1)
        y = y.view(n, -1)
        dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                 torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        dist_m.addmm_(1, -2, x, y.t())
        return dist_m

    @staticmethod
    def pairwise_distance_t2i(features, query=None, gallery=None):
        if query is None and gallery is None:
            n = len(features)
            x = torch.cat(list(features.values()))
            x = x.view(n, -1)
            dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
            dist_m = dist_m.expand(n, n) - 2 * torch.mm(x, x.t())
            return dist_m

        x = torch.cat([features[(f[0],f[1])].unsqueeze(0) for f in query], 0)
        y = torch.cat([features[(f[0],f[1])].unsqueeze(0) for f in gallery], 0)
        m, n = x.size(0), y.size(0)
        x = x.view(m, -1)
        y = y.view(n, -1)
        # import pdb;pdb.set_trace()
        dist_m = x @ y.t()
        
        # dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        # dist_m.addmm_(1, -2, x, y.t())
        
        return dist_m
    
    @staticmethod
    def pairwise_distance_t2i_f(features, features_bio, features_clot, attr_clot=None, query=None, gallery=None):
        if query is None and gallery is None:
            n = len(features)
            x = torch.cat(list(features.values()))
            x = x.view(n, -1)
            dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
            dist_m = dist_m.expand(n, n) - 2 * torch.mm(x, x.t())
            return dist_m
        x_f = torch.cat([features_clot[(f[0],f[1])].unsqueeze(0) for f in query], 0)
        y_f = torch.cat([features_bio[(f[0],f[1])].unsqueeze(0) for f in gallery], 0)
        if attr_clot is not None:
            x_f_attr = torch.cat([attr_clot[(f[0],f[1])].unsqueeze(0) for f in query], 0)
        else:
            x_f_attr = None
        
        x = torch.cat([features[(f[0],f[1])].unsqueeze(0) for f in query], 0)
        y = torch.cat([features[(f[0],f[1])].unsqueeze(0) for f in gallery], 0)
        m, n = x.size(0), y.size(0)
        x = x.view(m, -1)
        y = y.view(n, -1)
        
        # import pdb;pdb.set_trace()
        dist_m = x @ y.t()
        
        # dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        # dist_m.addmm_(1, -2, x, y.t())
        
        return dist_m, x_f, y_f, x_f_attr

    @staticmethod
    def extract_features(model, data_loader, validate_feat, print_freq=50):
        model.eval()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        features = OrderedDict()
        features_bio = OrderedDict()
        features_clot = OrderedDict()
        labels = OrderedDict()
        attr_clot = OrderedDict()

        end = time.time()
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                # img, clothes_img, fname, clothes_fname, pid, cid, cam, index
                imgs = data[0]
                clothes_imgs = data[1]
                fnames = data[2]
                clothes_fnames = data[3]
                pids = data[4]
                view_ids = data[5]
                cam_ids = data[6]

                data_time.update(time.time() - end)

                imgs = to_torch(imgs).cuda()
                # clothes_imgs = to_torch(clothes_imgs).cuda()
                cam_ids = cam_ids.cuda()
                view_ids = view_ids.cuda()
                # bio_feats, clot_feats, feat, clot_feats_s, clot_feats
                bio_feats, clot_feats, clo_attrs, outputs_img, text_features, text_features_n = model(imgs, clothes_imgs, cam_label=cam_ids, view_label=view_ids)
                if validate_feat == 'person':
                    outputs = bio_feats.data.cpu()
                elif validate_feat == 'clothes':
                    outputs = clot_feats.data.cpu()
                else:
                    outputs = outputs_img.data.cpu()
                    outputs_text_features = text_features.data.cpu()
                    outputs_bio_feats = bio_feats.data.cpu()
                    outputs_clot_feats = clot_feats.data.cpu()
                    output_clo_attrs = clo_attrs.data.cpu()

                for fname, clothes_fname, output, text_output, output_bio, output_clot, output_attr, pid in zip(fnames, clothes_fnames, outputs, outputs_text_features, outputs_bio_feats, outputs_clot_feats, output_clo_attrs, pids):
                    # import pdb;pdb.set_trace()
                    if clothes_fname == '001_1_c5_015874.png':
                        features[(fname,clothes_fname)] = output
                        features_bio[(fname,clothes_fname)] = output_bio
                    else:
                        features[(fname,clothes_fname)] = text_output
                        features_clot[(fname,clothes_fname)] = output_clot
                        attr_clot[(fname,clothes_fname)] = output_attr
                    labels[(fname,clothes_fname)] = pid

                batch_time.update(time.time() - end)
                end = time.time()

                if (i + 1) % print_freq == 0:
                    print('Extract Features: [{}/{}]\t'
                          'Time {:.3f} ({:.3f})\t'
                          'Data {:.3f} ({:.3f})\t'
                          .format(i + 1, len(data_loader),
                                  batch_time.val, batch_time.avg,
                                  data_time.val, data_time.avg))

        return features, labels, features_bio, features_clot, attr_clot

    @staticmethod
    def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50, ap_topk=500):
        """Evaluation with market1501 metric
            Key: for each query identity, its gallery images from the same camera view are discarded.
            """
        if (ap_topk is not None):
            assert (ap_topk >= max_rank)
        num_q, num_g = distmat.shape
        if num_g < max_rank:
            max_rank = num_g
            print("Note: number of gallery samples is quite small, got {}".format(num_g))
        # import pdb;pdb.set_trace()
        indices = np.argsort(-distmat, axis=1)
        matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
        # compute cmc curve for each query
        all_cmc = []
        all_AP = []
        num_valid_q = 0.  # number of valid query
        for q_idx in range(num_q):
            # get query pid and camid
            q_pid = q_pids[q_idx]
            q_camid = q_camids[q_idx]

            # remove gallery samples that have the same pid and camid with query
            order = indices[q_idx]
            remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
            keep = np.invert(remove)

            # compute cmc curve
            # binary vector, positions with value 1 are correct matches
            # orig_cmc = matches[q_idx][keep]

            if not np.any(matches[q_idx][keep]):
                # if not np.any(orig_cmc):
                # this condition is true when query identity does not appear in gallery
                continue

            if (ap_topk is None):
                orig_cmc = matches[q_idx][keep]
            else:
                orig_cmc = matches[q_idx][:ap_topk][keep[:ap_topk]]
                # orig_cmc = matches[q_idx][keep][:ap_topk]

            cmc = orig_cmc.cumsum()
            cmc[cmc > 1] = 1

            all_cmc.append(cmc[:max_rank])
            num_valid_q += 1.

            # compute average precision
            num_rel = orig_cmc.sum()
            tmp_cmc = orig_cmc.cumsum()
            tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
            tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
            AP = tmp_cmc.sum() / max(1, num_rel)
            all_AP.append(AP)

        assert num_valid_q > 0, "Error: all query identities do not appear in gallery"
        
        all_cmc = np.asarray(all_cmc).astype(np.float32)
        all_cmc = all_cmc.sum(0) / num_valid_q
        mAP = np.mean(all_AP)

        return all_cmc, mAP

