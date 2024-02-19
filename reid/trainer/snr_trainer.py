import torch.distributed as dist
from torch.nn import CrossEntropyLoss

from reid.loss import TripletLoss
from reid.loss.dual_causality_loss import DualCausalityLoss
from reid.trainer.base_trainer import BaseTrainer
from reid.utils import accuracy
from reid.utils.meters import AverageMeter


class SNRTrainer(BaseTrainer):
    def __init__(self, model, args):
        super(SNRTrainer, self).__init__(model, args)
        self.ce_loss = CrossEntropyLoss().cuda()
        self.triplet_loss = TripletLoss(margin=self.args.margin).cuda()
        self.dual_loss = DualCausalityLoss().cuda()

        self.losses_ce = AverageMeter()
        self.losses_tr = AverageMeter()
        self.losses_dual = AverageMeter()
        self.precisions = AverageMeter()

    def _logging(self, cur_iter):
        self._tensorboard_writer(cur_iter, data={
            'loss': self.losses_ce.val + self.losses_tr.val,
            'loss_ce': self.losses_ce.val,
            'loss_tr': self.losses_tr.val,
            'loss_dual': self.losses_dual.val,
            'prec': self.precisions.val
        })
        if not (cur_iter % self.args.print_freq == 0 and dist.get_rank() == 0):
            return
        print('Iter: [{}/{}]\t'
              'Time {:.3f} ({:.3f}) (ETA: {:.2f}h)\t'
              'Data {:.3f} ({:.3f})\t'
              'Loss_ce {:.3f} ({:.3f})\t'
              'Loss_tr {:.3f} ({:.3f})\t'
              'Loss_dual {:.3f} ({:.3f})\t'
              'Prec {:.2%} ({:.2%})'
              .format(cur_iter, self.args.iters,
                      self.batch_time.val, self.batch_time.avg,
                      (self.args.iters - cur_iter) * self.batch_time.avg / 3600,
                      self.data_time.val, self.data_time.avg,
                      self.losses_ce.val, self.losses_ce.avg,
                      self.losses_tr.val, self.losses_tr.avg,
                      self.losses_dual.val, self.losses_dual.avg,
                      self.precisions.val, self.precisions.avg))

    def _refresh_information(self, cur_iter, lr):
        if cur_iter % self.args.refresh_freq == 0 or cur_iter == 1:
            self.batch_time = AverageMeter()
            self.data_time = AverageMeter()
            self.losses_ce = AverageMeter()
            self.losses_tr = AverageMeter()
            self.losses_dual = AverageMeter()
            self.precisions = AverageMeter()
            if dist.get_rank() == 0:
                print("lr = {} \t".format(lr))

    def run(self, inputs):
        inputs, targets = self._parse_data(inputs)
        feat, _, logits, dual_list = self.model(inputs, targets)

        loss_ce = self.ce_loss(logits, targets)
        loss_tr, _ = self.triplet_loss(feat, targets)

        w = [0.1, 0.1, 0.5, 0.5]
        loss_dual = 0
        for idx, item in enumerate(dual_list):
            loss_dual += w[idx] * self.dual_loss(item, targets)

        loss = loss_ce + loss_tr + loss_dual

        self.losses_ce.update(loss_ce.item())
        self.losses_tr.update(loss_tr.item())
        self.losses_dual.update(loss_dual.item())

        prec, = accuracy(logits.data, targets.data)
        prec = prec[0]
        self.precisions.update(prec)

        return loss
