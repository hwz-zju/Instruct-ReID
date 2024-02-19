import torch.distributed as dist
from torch.nn import CrossEntropyLoss
import torch
from reid.loss import TripletLoss, TransLoss

from reid.trainer.base_trainer import BaseTrainer
from reid.utils import accuracy
from reid.utils.meters import AverageMeter


class TransreidTwobranchTrainer(BaseTrainer):
    def __init__(self, model, args, this_task_info=None):
        super(TransreidTwobranchTrainer, self).__init__(model, args, this_task_info)
        self.ce_loss = CrossEntropyLoss().cuda()
        self.triplet_loss = TripletLoss(margin=self.args.margin).cuda()
        if self.args.transe_loss:
            self.transe_loss = TransLoss(margin=self.args.margin).cuda() 
            self.losses_transe = AverageMeter() 

        self.losses_ce = AverageMeter()
        self.losses_bme = AverageMeter()
        self.losses_tr = AverageMeter()
        self.precisions = AverageMeter()
        self.loss_option = args.loss

    def _logging(self, cur_iter):
        if self.args.transe_loss:
            self._tensorboard_writer(cur_iter, data={
                'loss': self.losses_ce.val + self.losses_tr.val,
                'loss_ce': self.losses_ce.val,
                'loss_bme': self.losses_bme.val,
                'loss_tr': self.losses_tr.val,
                'loss_transe': self.losses_transe.val, 
                'prec': self.precisions.val
            })
        else:
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
        if self.args.transe_loss:
            print('Iter: [{}/{}]\t'
                'Time {:.3f} ({:.3f}) (ETA: {:.2f}h)\t'
                'Data {:.3f} ({:.3f})\t'
                'Loss_ce {:.3f} ({:.3f})\t'
                'Loss_tr {:.3f} ({:.3f})\t'
                'Loss_bme {:.3f} ({:.3f})\t'
                'Loss_transe {:.3f} ({:.3f})\t'
                'Prec {:.2%} ({:.2%})'
                .format(cur_iter, self.args.iters,
                        self.batch_time.val, self.batch_time.avg,
                        (self.args.iters - cur_iter) * self.batch_time.avg / 3600,
                        self.data_time.val, self.data_time.avg,
                        self.losses_ce.val, self.losses_ce.avg,
                        self.losses_tr.val, self.losses_tr.avg,
                        self.losses_bme.val, self.losses_bme.avg,
                        self.losses_transe.val, self.losses_transe.avg,
                        self.precisions.val, self.precisions.avg))
        else:
            print('Iter: [{}/{}]\t'
                'Time {:.3f} ({:.3f}) (ETA: {:.2f}h)\t'
                'Data {:.3f} ({:.3f})\t'
                'Loss_ce {:.3f} ({:.3f})\t'
                'Loss_tr {:.3f} ({:.3f})\t'
                'Loss_bme {:.3f} ({:.3f})\t'
                'Prec {:.2%} ({:.2%})'
                .format(cur_iter, self.args.iters,
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
            if self.args.transe_loss:
                self.losses_transe = AverageMeter() 
            local_rank = self.this_task_info.task_rank if self.this_task_info else dist.get_rank()
            if local_rank == 0:
                print("lr = {} \t".format(lr))

    def _parse_data(self, inputs):
        imgs, clothes, _, _, pids, _, _, indices = inputs
        inputs = imgs.cuda()
        clothes = clothes.cuda()
        targets = pids.cuda()
        return inputs, clothes, targets

    def run(self, inputs):
        inputs, clothes, targets = self._parse_data(inputs)
        
        if self.args.transe_loss:
            transe_inputs_pos, transe_clothes_pos = [], []
            transe_inputs_neg, transe_clothes_neg = [], []
            
            N = inputs.shape[0]
            label_mat = targets.expand(N, N).eq(targets.expand(N, N).t())
            for batch_idx in range(N):
                label = label_mat[batch_idx]
                select_pos_idxs = torch.where(label==1)[0]
                select_pos_idx = torch.randint(0, len(select_pos_idxs), (1,))
                transe_inputs_pos.append(inputs[[select_pos_idx]].clone())
                transe_clothes_pos.append(clothes[[batch_idx]].clone())
                select_neg_idxs = torch.where(label!=1)[0]
                select_neg_idx = torch.randint(0, len(select_neg_idxs), (1,))
                transe_inputs_neg.append(inputs[[select_neg_idx]].clone())
                transe_clothes_neg.append(clothes[[batch_idx]].clone())
            transe_inputs_pos = torch.cat(transe_inputs_pos, 0)
            transe_clothes_pos = torch.cat(transe_clothes_pos, 0)
            transe_inputs_neg = torch.cat(transe_inputs_neg, 0)
            transe_clothes_neg = torch.cat(transe_clothes_neg, 0)
            assert inputs.shape == transe_inputs_pos.shape


        logits, feat, logits_fusion, feat_fusion = self.model(inputs, clothes)

        if isinstance(logits_fusion, list):
            logits_global_fusion, logits_local_fusion = logits_fusion[0], logits_fusion[1:]
            
            if 'ce' in self.loss_option:
                loss_ce_global = self.ce_loss(logits_global_fusion, targets)
                loss_ce_local = [self.ce_loss(logit, targets) for logit in logits_local_fusion]
                
                loss_ce_local = sum(loss_ce_local)/len(loss_ce_local)
                loss_ce = 0.5*loss_ce_local + 0.5*loss_ce_global
            else:
                loss_ce = 0
            
        else:
            if 'ce' in self.loss_option:
                loss_ce = self.ce_loss(logits_fusion, targets)
            else:
                loss_ce = 0

        if isinstance(logits, list):
            if 'bio' in self.loss_option:
                logits_global_bme, logits_local_bme = logits[0], logits[1:]
                
                loss_ce_global_bme = self.ce_loss(logits_global_bme, targets)
                loss_ce_local_bme = [self.ce_loss(logit, targets) for logit in logits_local_bme]
                
                loss_ce_local_bme = sum(loss_ce_local_bme)/len(loss_ce_local_bme)
                loss_ce_bme = 0.5*loss_ce_local_bme + 0.5*loss_ce_global_bme
            else:
                loss_ce_bme = 0
            
        else:
            if 'bio' in self.loss_option:
                loss_ce_bme = self.ce_loss(logits, targets)
            else:
                loss_ce_bme = 0

        if isinstance(feat_fusion, list):
            if 'tri' in self.loss_option:
                feat_global_fusion, feat_local_fusion = feat_fusion[0], feat_fusion[1:]
                loss_tr_global_fusion, _ = self.triplet_loss(feat_global_fusion, targets)
                loss_tr_local_fusion = [self.triplet_loss(feat_, targets)[0] for feat_ in feat_local_fusion]
                loss_tr_local_fusion = sum(loss_tr_local_fusion)/len(loss_tr_local_fusion)
                loss_tr = 0.5*loss_tr_global_fusion + 0.5*loss_tr_local_fusion
            else:
                loss_tr = 0
        else:
            if 'tri' in self.loss_option:
                loss_tr, _ = self.triplet_loss(feat_fusion, targets)
            else:
                loss_tr = 0
        loss_tr = loss_tr 


        if self.args.transe_loss:
            
            logits1_pos, feat_pos, logits2_pos, fusion_feat_pos  = self.model(transe_inputs_pos, transe_clothes_pos)
            _, _, _, fusion_feat_neg = self.model(transe_inputs_neg, transe_clothes_neg)
            if isinstance(fusion_feat_pos, list):
                feat_global_fusion, feat_local_fusion = feat_fusion[0], feat_fusion[1:]
                fusion_feat_pos_global, fusion_feat_pos_local = fusion_feat_pos[0], fusion_feat_pos[1:]
                fusion_feat_neg_global, fusion_feat_neg_local = fusion_feat_neg[0], fusion_feat_neg[1:]
                loss_trans_global, _ = self.transe_loss(feat_global_fusion, fusion_feat_pos_global, fusion_feat_neg_global)
                loss_trans_local = [self.transe_loss(feat1_, feat2_, feat3_)[0] for feat1_, feat2_, feat3_ in zip(feat_local_fusion, fusion_feat_pos_local, fusion_feat_neg_local)]
                loss_trans_local = sum(loss_trans_local)/len(loss_trans_local)
                loss_trans = 0.5*loss_trans_global+0.5*loss_trans_local
            else:
                loss_trans, _ = self.transe_loss(feat_fusion, fusion_feat_pos, fusion_feat_neg)

            


        loss = loss_ce + loss_tr + loss_ce_bme
        if self.args.transe_loss:
            loss=loss+loss_trans*0.5
            self.losses_transe.update(loss_trans.item())

        if 'ce' in self.loss_option:
            self.losses_ce.update(loss_ce.item())
        if 'tri' in self.loss_option:
            self.losses_tr.update(loss_tr.item())
        if 'bio' in self.loss_option:
            self.losses_bme.update(loss_ce_bme.item())

        

        if isinstance(logits_fusion, list):
            prec, = accuracy(logits_fusion[0].data, targets.data) 
        else:
            prec, = accuracy(logits_fusion.data, targets.data) 
        prec = prec[0]
        
        self.precisions.update(prec)

        return loss
