import torch.distributed as dist
from torch.nn import CrossEntropyLoss

from reid.loss import TripletLoss
from reid.trainer.base_trainer_t2i import BaseTrainer
from reid.utils import accuracy
from reid.utils.meters import AverageMeter
import torch
import torch.nn.functional as F
# try:
#     import spring.linklink as link
# except:
#     import linklink as link
class GeneralTransformerTrainer_t2i(BaseTrainer):
    def __init__(self, model, args, this_task_info=None):
        super(GeneralTransformerTrainer_t2i, self).__init__(model, args, this_task_info)

        self.ce_loss = CrossEntropyLoss().cuda()
        self.triplet_loss = TripletLoss(margin=self.args.margin).cuda()

        self.losses_ce = AverageMeter()
        self.losses_bme = AverageMeter()
        self.losses_tr = AverageMeter()
        self.precisions = AverageMeter()

    def _logging(self, cur_iter):
        self._tensorboard_writer(cur_iter, data={
            'loss': self.losses_ce.val + self.losses_tr.val,
            'loss_ce': self.losses_ce.val,
            'loss_bme': self.losses_bme.val,
            'loss_tr': self.losses_tr.val,
            'prec': self.precisions.val
        })
        local_rank = self.this_task_info.task_rank if self.this_task_info else dist.get_rank()
        if not (cur_iter % self.args.print_freq == 0 and local_rank == 0):
            return
        if self.this_task_info:
            task_id, task_name = self.this_task_info.task_id, self.this_task_info.task_name
        else:
            task_id, task_name = 0, 'single task'
        print('Iter: [{}/{}]\t'
              'task{}: {}\t'
              'Time {:.3f} ({:.3f}) (ETA: {:.2f}h)\t'
              'Data {:.3f} ({:.3f})\t'
              'Loss_ce {:.3f} ({:.3f})\t'
              'Loss_tr {:.3f} ({:.3f})\t'
              'Loss_bme {:.3f} ({:.3f})\t'
              'Prec {:.2%} ({:.2%})'
              .format(cur_iter, self.args.iters,
                      str(task_id), str(task_name),
                      self.batch_time.val, self.batch_time.avg,
                      (self.args.iters - cur_iter) * self.batch_time.avg / 3600,
                      self.data_time.val, self.data_time.avg,
                      self.losses_ce.val, self.losses_ce.avg,
                      self.losses_tr.val, self.losses_tr.avg,
                      self.losses_bme.val, self.losses_bme.avg,
                      self.precisions.val, self.precisions.avg))

    def _refresh_information(self, cur_iter, lr):
        if cur_iter % self.args.refresh_freq == 0 or cur_iter == 1:
            self.batch_time = AverageMeter()
            self.data_time = AverageMeter()
            self.losses_ce = AverageMeter()
            self.losses_tr = AverageMeter()
            self.losses_bme = AverageMeter()
            self.precisions = AverageMeter()
            local_rank = self.this_task_info.task_rank if self.this_task_info else dist.get_rank()
            if local_rank == 0:
                print("lr = {} \t".format(lr))

    def _parse_data(self, inputs):
        imgs, instructions, _, _, pids, view_ids, cam_ids, indices = inputs
        inputs = imgs.cuda()
        # instructions = instructions.cuda()
        targets = pids.cuda()
        cam_ids = cam_ids.cuda()
        view_ids = view_ids.cuda()
        return inputs, instructions, targets, cam_ids, view_ids

    def run(self, inputs):
        inputs, instructions, targets, cam_ids, view_ids = self._parse_data(inputs)
        img_feat, text_feat, vl_f, vl_f_n, logits1, vl_output, vl_labels, loss_cl, loss_pitm = self.model(inputs, instructions, this_task_info=self.this_task_info, label=targets, cam_label=cam_ids, view_label=view_ids)
        
        
        # # idx = targets.view(-1, 1)
        # # pos_idx = torch.eq(idx, idx.t()).float()
        # # sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)
        
        # idx = targets.view(-1, 1)
        # idx_all = torch.cat([idx.t(), self.model.module.idx_queue.clone().detach()], dim=1)
        # # idx_all = torch.cat([idx.t()], dim=1)
        # pos_idx = torch.eq(idx, idx_all).float()
        # sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)
        
        # text_feat_all = torch.cat([text_feat.t(), self.model.module.text_queue.clone().detach()], dim=1)
        # # text_feat_all = torch.cat([text_feat.t()], dim=1)
        
        # image_feat_all = torch.cat([img_feat.t(), self.model.module.image_queue.clone().detach()], dim=1)
        # # image_feat_all = torch.cat([img_feat.t()], dim=1)
        # sim_i2t = img_feat @ text_feat_all / self.model.module.temp
        # sim_t2i = text_feat @ image_feat_all / self.model.module.temp
        # sim_i2i = img_feat @ image_feat_all / self.model.module.temp
        # sim_t2t = text_feat @ text_feat_all / self.model.module.temp
        
        
        # loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_targets, dim=1).mean()
        # loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_targets, dim=1).mean()
        # loss_i2i = -torch.sum(F.log_softmax(sim_i2i, dim=1) * sim_targets, dim=1).mean()
        # loss_t2t = -torch.sum(F.log_softmax(sim_t2t, dim=1) * sim_targets, dim=1).mean()
        # loss_cl = (loss_i2t + loss_t2i + loss_i2i + loss_t2t) / 4
        
        # self._dequeue_and_enqueue(img_feat, text_feat, idx)
        
        if self.args.fusion_loss=='all':
            if isinstance(logits1, list):
                ID_LOSS = [self.ce_loss(scor, targets) for scor in logits1[1:]]
                ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                loss_ce_biometric = 0.5 * ID_LOSS + 0.5 * self.ce_loss(logits1[0], targets)
            else:
                loss_ce_biometric = self.ce_loss(logits1, targets)
        # import pdb;pdb.set_trace()
        # vl_output = torch.cat([p_logits, n_logits], dim=0)
        # vl_labels = torch.cat([torch.ones(logits1.shape[0], dtype=torch.long), torch.zeros(2 * logits1.shape[0], dtype=torch.long)], dim=0).to(vl_output.device)
        # vl_labels = torch.cat([torch.ones(logits1.shape[0], dtype=torch.long)], dim=0).to(vl_output.device)
        # loss_ce_vl = self.ce_loss(vl_output, vl_labels)
        # loss_ce_vl = F.cross_entropy(vl_output, vl_labels)
        loss_ce_vl = loss_pitm
        
        # targets_p = torch.ones(p_logits.shape[0], dtype=torch.long).to(p_logits.device)
        # F.cross_entropy(p_logits, targets_p)
        # loss_ce_p = self.ce_loss(p_logits, targets_p)
        # targets_n = torch.zeros(n_logits.shape, dtype=torch.long).to(n_logits.device)
        # loss_ce_n = self.ce_loss(n_logits, targets_n)
        # if isinstance(feat, list):
        #     TRI_LOSS = [self.triplet_loss(feats, targets, clot_feats_s)[0] for feats in feat[1:]]
        #     TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
        #     loss_tr_biometric = 0.5 * TRI_LOSS + 0.5 * self.triplet_loss(feat[0], targets, clot_feats_s)[0]
        # else:
        #     loss_tr_biometric = self.triplet_loss(feat, targets, clot_feats_s)[0]
        loss_ce = 0
        loss_tr = 0
        # if 'bio' in self.args.fusion_branch:
        #     if isinstance(logits2, list):
        #         ID_LOSS = [self.ce_loss(scor, targets) for scor in logits2[1:]]
        #         ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
        #         loss_ce += 0.5 * ID_LOSS + 0.5 * self.ce_loss(logits2[0], targets)
        #     else:
        #         loss_ce+=self.ce_loss(logits2, targets)
        #     if isinstance(bio_f, list):
        #         TRI_LOSS = [self.triplet_loss(feats, targets, clot_feats_s)[0] for feats in bio_f[1:]]
        #         TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
        #         loss_tr += 0.5 * TRI_LOSS + 0.5 * self.triplet_loss(bio_f[0], targets, clot_feats_s)[0]
        #     else:
        #         loss_tr+=self.triplet_loss(bio_f, targets, clot_feats_s)[0]
        #         # loss_tr+=self.triplet_loss(bio_f, targets)[0]
        #         # loss_tr+=self.triplet_loss(bio_f_p, targets, clot_feats_s)[0]
        # if 'clot' in self.args.fusion_branch:
        #     if isinstance(logits3, list):
        #         ID_LOSS = [self.ce_loss(scor, targets) for scor in logits3[1:]]
        #         ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
        #         loss_ce += 0.5 * ID_LOSS + 0.5 * self.ce_loss(logits3[0], targets)
        #     else:
        #         loss_ce+=self.ce_loss(logits3, targets)
        #     if isinstance(clot_f, list):
        #         TRI_LOSS = [self.triplet_loss(feats, targets, clot_feats_s)[0] for feats in clot_f[1:]]
        #         TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
        #         loss_tr += 0.5 * TRI_LOSS + 0.5 * self.triplet_loss(clot_f[0], targets, clot_feats_s)[0]
        #     else:
        #         # loss_tr+=self.triplet_loss(clot_f, targets)[0]
        #         loss_tr+=self.triplet_loss(clot_f, targets, clot_feats_s)[0]
        #loss_ce = self.ce_loss(logits2, targets)
        #loss_ce += self.ce_loss(logits3, targets)
        ## loss_tr, _ = self.triplet_loss(feat, targets)
        

        #loss_tr, _ = self.triplet_loss(bio_f, targets)
        #loss_tr += self.triplet_loss(clot_f, targets)[0]
        # import pdb;pdb.set_trace()
        ratio_rate = 0.0
        if self.args.fusion_loss=='all':
            if 'bio' in self.args.fusion_branch and 'clot' in self.args.fusion_branch:
                # loss = ratio_rate * loss_ce / 2 + ratio_rate * self.args.alpha * loss_tr / 2 + loss_ce_biometric + self.args.alpha * loss_tr_biometric
                loss = 0.5*loss_cl + 0*loss_ce_biometric + loss_ce_vl
                # loss = ratio_rate * loss_ce / 2 + ratio_rate * self.args.alpha * loss_tr / 2 + loss_ce_biometric
            else:
                loss = loss_ce + self.args.alpha * loss_tr + loss_ce_biometric + loss_tr_biometric
        else:
            if 'bio' in self.args.fusion_branch and 'clot' in self.args.fusion_branch:
                loss = loss_ce / 2 + self.args.alpha * loss_tr / 2  
            else:
                loss = loss_ce + self.args.alpha * loss_tr 
        # import pdb;pdb.set_trace()
        self.losses_ce.update(loss_ce_vl.item())
        # self.losses_ce.update(self.ce_loss(logits2, targets).item())
        self.losses_tr.update(loss_cl.item())
        if self.args.fusion_loss=='all':
            self.losses_bme.update(loss_ce_biometric.item())
            # self.losses_bme.update(self.ce_loss(logits3, targets).item())
        if 'bio' in self.args.fusion_branch:
            if isinstance(logits1, list):
                prec, = accuracy(logits1[0].data, targets.data)
            else:
                # prec, = accuracy(logits1.data, targets.data)
                prec, = accuracy(vl_output.data, vl_labels.data)
        else:
            if isinstance(logits3, list):
                prec, = accuracy(logits3[0].data, targets.data)
            else:
                prec, = accuracy(logits3.data, targets.data)
        prec = prec[0]
        self.precisions.update(prec)

        return loss
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat, idx):
        # gather keys before updating queue
        # image_feats = concat_all_gather(image_feat)
        # text_feats = concat_all_gather(text_feat)
        # idxs = concat_all_gather(idx)
        image_feats = image_feat
        text_feats = text_feat
        idxs = idx
        batch_size = image_feats.shape[0]
        ptr = int(self.model.module.queue_ptr)
        # replace the keys at ptr (dequeue and enqueue)
        empty = self.model.module.image_queue.size(1) - ptr
        if batch_size <= empty:
            self.model.module.image_queue[:, ptr:ptr + batch_size] = image_feats.T
            self.model.module.text_queue[:, ptr:ptr + batch_size] = text_feats.T
            self.model.module.idx_queue[:, ptr:ptr + batch_size] = idxs.T
        else:
            self.model.module.image_queue[:, ptr:] = image_feats[:empty].T
            self.model.module.text_queue[:, ptr:] = text_feats[:empty].T
            self.model.module.idx_queue[:, ptr:] = idxs[:empty].T
            self.model.module.image_queue[:, :batch_size - empty] = image_feats[empty:].T
            self.model.module.text_queue[:, :batch_size - empty] = text_feats[empty:].T
            self.model.module.idx_queue[:, :batch_size - empty] = idxs[empty:].T
        ptr = (ptr + batch_size) % 65536  # move pointer
        self.model.module.queue_ptr[0] = ptr

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(link.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output