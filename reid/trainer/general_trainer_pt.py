import torch.distributed as dist
from torch.nn import CrossEntropyLoss

from reid.loss import TripletLoss
from reid.trainer.base_trainer import BaseTrainer
from reid.utils import accuracy
from reid.utils.meters import AverageMeter


class GeneralTrainer(BaseTrainer):
    def __init__(self, model, args, this_task_info=None):
        super(GeneralTrainer, self).__init__(model, args, this_task_info)
        self.ce_loss = CrossEntropyLoss().cuda()
        self.triplet_loss = TripletLoss(margin=self.args.margin).cuda()

        self.losses_ce = AverageMeter()
        self.losses_tr = AverageMeter()
        self.precisions = AverageMeter()

    def _logging(self, cur_iter):
        self._tensorboard_writer(cur_iter, data={
            'loss': self.losses_ce.val + self.losses_tr.val,
            'loss_ce': self.losses_ce.val,
            'loss_tr': self.losses_tr.val,
            'prec': self.precisions.val
        })

        if not (cur_iter % self.args.print_freq == 0 and dist.get_rank() == 0):
            return
        print('Iter: [{}/{}]\t'
              'Time {:.3f} ({:.3f}) (ETA: {:.2f}h)\t'
              'Data {:.3f} ({:.3f})\t'
              'Loss_ce {:.3f} ({:.3f})\t'
              'Loss_tr {:.3f} ({:.3f})\t'
              'Prec {:.2%} ({:.2%})'
              .format(cur_iter, self.args.iters,
                      self.batch_time.val, self.batch_time.avg,
                      (self.args.iters - cur_iter) * self.batch_time.avg / 3600,
                      self.data_time.val, self.data_time.avg,
                      self.losses_ce.val, self.losses_ce.avg,
                      self.losses_tr.val, self.losses_tr.avg,
                      self.precisions.val, self.precisions.avg))

    def _refresh_information(self, cur_iter, lr):
        if cur_iter % self.args.refresh_freq == 0 or cur_iter == 1:
            self.batch_time = AverageMeter()
            self.data_time = AverageMeter()
            self.losses_ce = AverageMeter()
            self.losses_tr = AverageMeter()
            self.precisions = AverageMeter()
            if dist.get_rank() == 0:
                print("lr = {} \t".format(lr))
    def _parse_data(self, inputs):
        imgs, clothes, _, _, pids, _, _, indices = inputs
        inputs = imgs.cuda()
        clothes = clothes.cuda()
        targets = pids.cuda()
        return inputs, clothes, targets

    def run(self, inputs):
        inputs, clothes, targets = self._parse_data(inputs)
        feat, logits, clot_feats_s = self.model(inputs, clothes)

        loss_ce = self.ce_loss(logits, targets)
        loss_tr, _ = self.triplet_loss(feat, targets, clot_feats_s)
        # import pdb;pdb.set_trace()
        # loss_tr, _ = self.triplet_loss(feat, targets)
        loss = loss_ce + loss_tr

        self.losses_ce.update(loss_ce.item())
        self.losses_tr.update(loss_tr.item())

        prec, = accuracy(logits.data, targets.data)
        prec = prec[0]
        self.precisions.update(prec)

        return loss
