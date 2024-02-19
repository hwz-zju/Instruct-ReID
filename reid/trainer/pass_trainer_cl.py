import torch.distributed as dist
from torch.nn import CrossEntropyLoss

from reid.loss import TripletLoss
from reid.trainer.base_trainer_t2i import BaseTrainer
from reid.utils import accuracy
from reid.utils.meters import AverageMeter
import torch
import torch.nn.functional as F

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
        instructions = instructions.cuda()
        targets = pids.cuda()
        cam_ids = cam_ids.cuda()
        view_ids = view_ids.cuda()
        return inputs, instructions, targets, cam_ids, view_ids

    def run(self, inputs):
        inputs, instructions, targets, cam_ids, view_ids = self._parse_data(inputs)
        # feat, vl_f, vl_f_n, logits1, p_logits, n_logits, clot_feats_s
        feat, vl_f, vl_f_n, logits1, vl_output, clot_feats_s = self.model(inputs, instructions, this_task_info=self.this_task_info, label=targets, cam_label=cam_ids, view_label=view_ids)
        
        
        idx = targets.view(-1, 1)
        pos_idx = torch.eq(idx, idx.t()).float()
        sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)
        
        # #这步得到它的不同类的矩阵，不同类的位置为1
        # sim_targets_no_sim = torch.ones_like(pos_idx) - pos_idx
        # sim_targets_no_sim = sim_targets_no_sim / sim_targets_no_sim.sum(1, keepdim=True)
        
        # import pdb; pdb.set_trace()
        
        sim_i2t = feat @ clot_feats_s.t()
        sim_t2i = clot_feats_s @ feat.t()
        sim_i2i = feat @ feat.t()
        sim_t2t = clot_feats_s @ clot_feats_s.t()
        
        # sim_i2t = torch.exp(sim_i2t)
        # sim_t2i = torch.exp(sim_t2i)
        # sim_i2i = torch.exp(sim_i2i)
        # sim_t2t = torch.exp(sim_t2t)
        
        # sim_i2t_p = pos_idx * sim_i2t
        # sim_t2i_p = pos_idx * sim_t2i
        # sim_i2i_p = pos_idx * sim_i2i
        # sim_t2t_p = pos_idx * sim_t2t
        
        # sim_i2t_n = sim_i2t - sim_i2t_p
        # sim_t2i_n = sim_t2i - sim_t2i_p
        # sim_i2i_n = sim_i2i - sim_i2i_p
        # sim_t2t_n = sim_t2t - sim_t2t_p
        
        # sim_i2t_n_sum = torch.sum(sim_i2t_n , dim=1)
        # sim_t2i_n_sum = torch.sum(sim_t2i_n , dim=1)
        # sim_i2i_n_sum = torch.sum(sim_i2i_n , dim=1)
        # sim_t2t_n_sum = torch.sum(sim_t2t_n , dim=1)
        
        # sim + no_sim_sum_expend
        
        
        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_targets, dim=1).mean()
        loss_i2i = -torch.sum(F.log_softmax(sim_i2i, dim=1) * sim_targets, dim=1).mean()
        loss_t2t = -torch.sum(F.log_softmax(sim_t2t, dim=1) * sim_targets, dim=1).mean()
        loss_cl = (loss_i2t + loss_t2i + loss_i2i + loss_t2t) / 4
        
        # import pdb;pdb.set_trace()
        
        # #这步得到它的label矩阵，相同label的位置为1
        # mask = torch.ones_like(similarity_matrix) * (label.expand(n, n).eq(label.expand(n, n).t()))

        # #这步得到它的不同类的矩阵，不同类的位置为1
        # mask_no_sim = torch.ones_like(mask) - mask

        # #这步产生一个对角线全为0的，其他位置为1的矩阵
        # mask_dui_jiao_0 = torch.ones(n ,n) - torch.eye(n, n )

        # #这步给相似度矩阵求exp,并且除以温度参数T
        # similarity_matrix = torch.exp(similarity_matrix/T)

        # #这步将相似度矩阵的对角线上的值全置0，因为对比损失不需要自己与自己的相似度
        # similarity_matrix = similarity_matrix*mask_dui_jiao_0


        # #这步产生了相同类别的相似度矩阵，标签相同的位置保存它们的相似度，其他位置都是0，对角线上也为0
        # sim = mask*similarity_matrix


        # #用原先的对角线为0的相似度矩阵减去相同类别的相似度矩阵就是不同类别的相似度矩阵
        # no_sim = similarity_matrix - sim


        # #把不同类别的相似度矩阵按行求和，得到的是对比损失的分母(还差一个与分子相同的那个相似度，后面会加上)
        # no_sim_sum = torch.sum(no_sim , dim=1)

        # '''
        # 将上面的矩阵扩展一下，再转置，加到sim（也就是相同标签的矩阵上），然后再把sim矩阵与sim_num矩阵做除法。
        # 至于为什么这么做，就是因为对比损失的分母存在一个同类别的相似度，就是分子的数据。做了除法之后，就能得到
        # 每个标签相同的相似度与它不同标签的相似度的值，它们在一个矩阵（loss矩阵）中。
        # '''
        # no_sim_sum_expend = no_sim_sum.repeat(n, 1).T
        # sim_sum  = sim + no_sim_sum_expend
        # loss = torch.div(sim , sim_sum)


        # '''
        # 由于loss矩阵中，存在0数值，那么在求-log的时候会出错。这时候，我们就将loss矩阵里面为0的地方
        # 全部加上1，然后再去求loss矩阵的值，那么-log1 = 0 ，就是我们想要的。
        # '''
        # loss = mask_no_sim + loss + torch.eye(n, n )


        # #接下来就是算一个批次中的loss了
        # loss = -torch.log(loss)  #求-log
        # loss = torch.sum(torch.sum(loss, dim=1) )/(2*n)  #将所有数据都加起来除以2n

        
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
                loss = 0*loss_cl + loss_ce_biometric
                # loss = ratio_rate * loss_ce / 2 + ratio_rate * self.args.alpha * loss_tr / 2 + loss_ce_biometric
            else:
                loss = loss_ce + self.args.alpha * loss_tr + loss_ce_biometric + loss_tr_biometric
        else:
            if 'bio' in self.args.fusion_branch and 'clot' in self.args.fusion_branch:
                loss = loss_ce / 2 + self.args.alpha * loss_tr / 2  
            else:
                loss = loss_ce + self.args.alpha * loss_tr 
        # import pdb;pdb.set_trace()
        self.losses_ce.update(loss_ce_biometric.item())
        # self.losses_ce.update(self.ce_loss(logits2, targets).item())
        self.losses_tr.update(loss_cl.item())
        if self.args.fusion_loss=='all':
            self.losses_bme.update(loss_ce_biometric.item())
            # self.losses_bme.update(self.ce_loss(logits3, targets).item())
        if 'bio' in self.args.fusion_branch:
            if isinstance(logits1, list):
                prec, = accuracy(logits1[0].data, targets.data)
            else:
                prec, = accuracy(logits1.data, targets.data)
        else:
            if isinstance(logits3, list):
                prec, = accuracy(logits3[0].data, targets.data)
            else:
                prec, = accuracy(logits3.data, targets.data)
        prec = prec[0]
        self.precisions.update(prec)

        return loss