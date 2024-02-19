import torch.distributed as dist
from torch.nn import CrossEntropyLoss

# from reid.loss import TripletLoss
# from reid.loss.triplet_transreid import TripletLoss
from reid.loss.triplet import TripletLoss
from reid.trainer.base_trainer import BaseTrainer
from reid.utils import accuracy
from reid.utils.meters import AverageMeter
# try:
#     import spring.linklink as link
# except:
#     import linklink as link


class GeneralTrainer(BaseTrainer):
    def __init__(self, model, args, this_task_info=None):
        super(GeneralTrainer, self).__init__(model, args, this_task_info)
        self.ce_loss = CrossEntropyLoss().cuda()
        self.triplet_loss = TripletLoss(margin=self.args.margin).cuda()
        # self.triplet_loss = TripletLoss().cuda()

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

        # if not (cur_iter % self.args.print_freq == 0 and dist.get_rank() == 0):
        #     return
        # print('Iter: [{}/{}]\t'
        #       'Time {:.3f} ({:.3f}) (ETA: {:.2f}h)\t'
        #       'Data {:.3f} ({:.3f})\t'
        #       'Loss_ce {:.3f} ({:.3f})\t'
        #       'Loss_tr {:.3f} ({:.3f})\t'
        #       'Prec {:.2%} ({:.2%})'
        #       .format(cur_iter, self.args.iters,
        #               self.batch_time.val, self.batch_time.avg,
        #               (self.args.iters - cur_iter) * self.batch_time.avg / 3600,
        #               self.data_time.val, self.data_time.avg,
        #               self.losses_ce.val, self.losses_ce.avg,
        #               self.losses_tr.val, self.losses_tr.avg,
        #               self.precisions.val, self.precisions.avg))

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
        # if cur_iter % self.args.refresh_freq == 0 or cur_iter == 1:
        #     self.batch_time = AverageMeter()
        #     self.data_time = AverageMeter()
        #     self.losses_ce = AverageMeter()
        #     self.losses_tr = AverageMeter()
        #     self.precisions = AverageMeter()
        #     if dist.get_rank() == 0:
        #         print("lr = {} \t".format(lr))
    def _parse_data(self, inputs):
        # img, clothes_img, fname, clothes_fname, pid, cid, cam, index
        # imgs, _, _, _, pids, _, _, indices = inputs
        # inputs = imgs.cuda()
        # targets = pids.cuda()
        # return inputs, targets
        imgs, clothes, _, _, pids, view_ids, cam_ids, indices = inputs
        inputs = imgs.cuda()
        clothes = clothes.cuda()
        targets = pids.cuda()
        cam_ids = cam_ids.cuda()
        view_ids = view_ids.cuda()
        return inputs, clothes, targets, cam_ids, view_ids

    def run(self, inputs):
        # inputs, targets = self._parse_data(inputs)
        # feat, _, logits = self.model(inputs, targets)
        inputs, clothes, targets, cam_ids, view_ids = self._parse_data(inputs)
        feat, logits, clot_feats_s = self.model(inputs, clothes, cam_label=cam_ids, view_label=view_ids)

        if self.args.fusion_loss=='all':
            if isinstance(logits, list):
                ID_LOSS = [self.ce_loss(scor, targets) for scor in logits[1:]]
                ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                loss_ce = 0.5 * ID_LOSS + 0.5 * self.ce_loss(logits[0], targets)
            else:
                loss_ce = self.ce_loss(logits, targets)
        if isinstance(feat, list):
            TRI_LOSS = [self.triplet_loss(feats, targets)[0] for feats in feat[1:]]
            TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
            loss_tr = 0.5 * TRI_LOSS + 0.5 * self.triplet_loss(feat[0], targets)[0]
        else:
            loss_tr = self.triplet_loss(feat, targets)[0]
        
        # loss_ce = self.ce_loss(logits, targets)
        # # loss_tr, _ = self.triplet_loss(feat, targets, clot_feats_s)
        # # import pdb;pdb.set_trace()
        # loss_tr, _, _ = self.triplet_loss(feat, targets)
        loss = loss_ce + loss_tr

        self.losses_ce.update(loss_ce.item())
        self.losses_tr.update(loss_tr.item())

        # prec, = accuracy(logits.data, targets.data)
        if isinstance(logits, list):
            prec, = accuracy(logits[0].data, targets.data)
        else:
            prec, = accuracy(logits.data, targets.data)
        prec = prec[0]
        self.precisions.update(prec)

        return loss
