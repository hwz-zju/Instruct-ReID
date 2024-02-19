import torch.distributed as dist
from torch.nn import CrossEntropyLoss

from reid.loss import TripletLoss
# from reid.loss.triplet import TripletLoss
from reid.trainer.base_trainer import BaseTrainer
from reid.utils import accuracy
from reid.utils.meters import AverageMeter

# import linklink as link
# try:
#     import spring.linklink as link
# except:
#     import linklink as link


class GeneralTransformerTrainer_attr(BaseTrainer):
    def __init__(self, model, args, this_task_info=None):
        super(GeneralTransformerTrainer_attr, self).__init__(model, args, this_task_info)

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
            local_rank = link.get_rank() if self.this_task_info else dist.get_rank()
            if local_rank == 0:
                print("lr = {} \t".format(lr))

    def _parse_data(self, inputs):
        # imgs, clothes, _, _, pids, _, _, indices = inputs
        imgs, attrs, _, _, pids, _, _, indices = inputs
        # import pdb;pdb.set_trace()
        inputs = imgs.cuda()
        # clothes = clothes.cuda()
        attrs = attrs.cuda()
        targets = pids.cuda()
        return inputs, attrs, targets

    def run(self, inputs):
        inputs, attrs, targets = self._parse_data(inputs)
        # feat, bio_f, bio_f_p, clot_f, clot_f_p, logits1, logits2, logits3, clot_feats_s = self.model(inputs, attrs, self.this_task_info)
        feat, bio_f, clot_f, logits1, logits2, logits3, clot_feats_s = self.model(inputs, attrs, self.this_task_info)
        if self.args.fusion_loss=='all':
            loss_ce_biometric = self.ce_loss(logits1, targets)
        loss_tr_biometric = self.triplet_loss(feat, targets, clot_feats_s)[0]

        loss_ce = 0
        loss_tr = 0
        if 'bio' in self.args.fusion_branch:
            loss_ce+=self.ce_loss(logits2, targets)
            loss_tr+=self.triplet_loss(bio_f, targets, clot_feats_s)[0]
            # loss_tr+=self.triplet_loss(bio_f_p, targets, clot_feats_s)[0]
        if 'clot' in self.args.fusion_branch:
            loss_ce+=self.ce_loss(logits3, targets)
            loss_tr+= self.triplet_loss(clot_f, targets, clot_feats_s)[0]
            # loss_tr+= self.triplet_loss(clot_f_p, targets, clot_feats_s)[0]
        #loss_ce = self.ce_loss(logits2, targets)
        #loss_ce += self.ce_loss(logits3, targets)
        ## loss_tr, _ = self.triplet_loss(feat, targets)
        

        #loss_tr, _ = self.triplet_loss(bio_f, targets)
        #loss_tr += self.triplet_loss(clot_f, targets)[0]
        # import pdb;pdb.set_trace()
        if self.args.fusion_loss=='all':
            if 'bio' in self.args.fusion_branch and 'clot' in self.args.fusion_branch:
                loss = loss_ce / 2 + self.args.alpha * loss_tr / 2 + loss_ce_biometric + self.args.alpha * loss_tr_biometric
            else:
                loss = loss_ce + self.args.alpha * loss_tr + loss_ce_biometric + loss_tr_biometric
        else:
            if 'bio' in self.args.fusion_branch and 'clot' in self.args.fusion_branch:
                loss = loss_ce / 2 + self.args.alpha * loss_tr / 2  
            else:
                loss = loss_ce + self.args.alpha * loss_tr 

        # self.losses_ce.update(loss_ce.item())
        self.losses_ce.update(self.ce_loss(logits2, targets).item())
        self.losses_tr.update(loss_tr.item())
        if self.args.fusion_loss=='all':
            # self.losses_bme.update(loss_ce_biometric.item())
            self.losses_bme.update(self.ce_loss(logits3, targets).item())
        if 'bio' in self.args.fusion_branch:
            prec, = accuracy(logits2.data, targets.data)
        else:
            prec, = accuracy(logits3.data, targets.data)
        prec = prec[0]
        self.precisions.update(prec)

        return loss
