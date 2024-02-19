import torch.distributed as dist
from torch.nn import CrossEntropyLoss

from reid.loss import TripletLoss
from reid.trainer.base_trainer import BaseTrainer
from reid.utils import accuracy
from reid.utils.meters import AverageMeter

class GeneralTransformerTrainer(BaseTrainer):
    def __init__(self, model, args, this_task_info=None):
        super(GeneralTransformerTrainer, self).__init__(model, args, this_task_info)

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
        feat, bio_f, clot_f, logits1, logits2, logits3, clot_feats_s = self.model(inputs, instructions, this_task_info=self.this_task_info, cam_label=cam_ids, view_label=view_ids)
        if self.args.fusion_loss=='all':
            if isinstance(logits1, list):
                ID_LOSS = [self.ce_loss(scor, targets) for scor in logits1[1:]]
                ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                loss_ce_biometric = 0.5 * ID_LOSS + 0.5 * self.ce_loss(logits1[0], targets)
            else:
                loss_ce_biometric = self.ce_loss(logits1, targets)
        if isinstance(feat, list):
            TRI_LOSS = [self.triplet_loss(feats, targets, clot_feats_s)[0] for feats in feat[1:]]
            TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
            loss_tr_biometric = 0.5 * TRI_LOSS + 0.5 * self.triplet_loss(feat[0], targets, clot_feats_s)[0]
        else:
            loss_tr_biometric = self.triplet_loss(feat, targets, clot_feats_s)[0]
        loss_ce = 0
        loss_tr = 0
        if 'bio' in self.args.fusion_branch:
            if isinstance(logits2, list):
                ID_LOSS = [self.ce_loss(scor, targets) for scor in logits2[1:]]
                ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                loss_ce += 0.5 * ID_LOSS + 0.5 * self.ce_loss(logits2[0], targets)
            else:
                loss_ce+=self.ce_loss(logits2, targets)
            if isinstance(bio_f, list):
                TRI_LOSS = [self.triplet_loss(feats, targets, clot_feats_s)[0] for feats in bio_f[1:]]
                TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                loss_tr += 0.5 * TRI_LOSS + 0.5 * self.triplet_loss(bio_f[0], targets, clot_feats_s)[0]
            else:
                loss_tr+=self.triplet_loss(bio_f, targets, clot_feats_s)[0]
                # loss_tr+=self.triplet_loss(bio_f, targets)[0]
                # loss_tr+=self.triplet_loss(bio_f_p, targets, clot_feats_s)[0]
        if 'clot' in self.args.fusion_branch:
            if isinstance(logits3, list):
                ID_LOSS = [self.ce_loss(scor, targets) for scor in logits3[1:]]
                ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                loss_ce += 0.5 * ID_LOSS + 0.5 * self.ce_loss(logits3[0], targets)
            else:
                loss_ce+=self.ce_loss(logits3, targets)
            if isinstance(clot_f, list):
                TRI_LOSS = [self.triplet_loss(feats, targets, clot_feats_s)[0] for feats in clot_f[1:]]
                TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                loss_tr += 0.5 * TRI_LOSS + 0.5 * self.triplet_loss(clot_f[0], targets, clot_feats_s)[0]
            else:
                # loss_tr+=self.triplet_loss(clot_f, targets)[0]
                loss_tr+=self.triplet_loss(clot_f, targets, clot_feats_s)[0]
        #loss_ce = self.ce_loss(logits2, targets)
        #loss_ce += self.ce_loss(logits3, targets)
        ## loss_tr, _ = self.triplet_loss(feat, targets)
        

        #loss_tr, _ = self.triplet_loss(bio_f, targets)
        #loss_tr += self.triplet_loss(clot_f, targets)[0]
        # import pdb;pdb.set_trace()
        ratio_rate = 0.1
        if self.args.fusion_loss=='all':
            if 'bio' in self.args.fusion_branch and 'clot' in self.args.fusion_branch:
                loss = ratio_rate * loss_ce / 2 + ratio_rate * self.args.alpha * loss_tr / 2 + loss_ce_biometric + self.args.alpha * loss_tr_biometric
                # loss = ratio_rate * loss_ce / 2 + ratio_rate * self.args.alpha * loss_tr / 2 + loss_ce_biometric
            else:
                loss = loss_ce + self.args.alpha * loss_tr + loss_ce_biometric + loss_tr_biometric
        else:
            if 'bio' in self.args.fusion_branch and 'clot' in self.args.fusion_branch:
                loss = loss_ce / 2 + self.args.alpha * loss_tr / 2  
            else:
                loss = loss_ce + self.args.alpha * loss_tr 
        # import pdb;pdb.set_trace()
        self.losses_ce.update(loss_ce.item())
        # self.losses_ce.update(self.ce_loss(logits2, targets).item())
        self.losses_tr.update(loss_tr.item())
        if self.args.fusion_loss=='all':
            self.losses_bme.update(loss_ce_biometric.item())
            # self.losses_bme.update(self.ce_loss(logits3, targets).item())
        if 'bio' in self.args.fusion_branch:
            if isinstance(logits2, list):
                prec, = accuracy(logits2[0].data, targets.data)
            else:
                prec, = accuracy(logits2.data, targets.data)
        else:
            if isinstance(logits3, list):
                prec, = accuracy(logits3[0].data, targets.data)
            else:
                prec, = accuracy(logits3.data, targets.data)
        prec = prec[0]
        self.precisions.update(prec)

        return loss