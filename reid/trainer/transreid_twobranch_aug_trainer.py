import torch.distributed as dist
from torch.nn import CrossEntropyLoss
import torch
import torchvision.transforms.functional_tensor as Ft
import torchvision.transforms as T
torch.autograd.set_detect_anomaly(True)
import torch.nn.functional as F
from reid.loss import TripletLoss, TransLoss

from reid.trainer.base_trainer import BaseTrainer
from reid.utils import accuracy, osutils
from reid.utils.meters import AverageMeter
import time
import os.path as osp
import time

from reid.evaluation.evaluators import Evaluator
from reid.utils.meters import AverageMeter
from reid.utils.serialization import save_checkpoint
from tensorboardX.writer import SummaryWriter

from reid.multi_tasks_utils.multi_task_distributed_utils import multitask_reduce_gradients

def set_grad(nets, requires_grad=False):
    for net in nets:
        for param in net.parameters():
            param.requires_grad = requires_grad


class TransreidTwobranchAugTrainer(BaseTrainer):
    def __init__(self, model, model_aug_bio, model_aug_clo, args, this_task_info=None):
        super(TransreidTwobranchAugTrainer, self).__init__(model, args, this_task_info=this_task_info)
        self.model_aug_bio = model_aug_bio
        self.model_aug_clo = model_aug_clo
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
        local_rank = self.this_task_info.task_rank if self.this_task_info is not None else dist.get_rank()
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
            if self.this_task_info:
                task_id, task_name =  self.this_task_info.task_id, self.this_task_info.task_name
            else:
                task_id, task_name = 0, 'single_task'
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
            if self.args.transe_loss:
                self.losses_transe = AverageMeter() 
            local_rank = self.this_task_info.task_rank if self.this_task_info is not None else dist.get_rank()
            if local_rank == 0:
                print("lr = {} \t".format(lr))

    def _parse_data(self, inputs):
        imgs, clothes, _, _, pids, _, _, indices = inputs
        inputs = imgs.cuda()
        clothes = clothes.cuda()
        targets = pids.cuda()
        return inputs, clothes, targets


    def train(self, data_loader, data_loader_aug, optimizer, optimizer_aug_bio, optimizer_aug_clo, lr_scheduler, lr_scheduler_bio, lr_scheduler_clo, test_loader=None, query=None, gallery=None,aug_start_iter=0):
        self.model.train()
        self.model_aug_bio.train()
        self.model_aug_clo.train()
        end = time.time()
        best_mAP, best_iter = 0, 0
        current_iter = 0
        for inputs, inputs_for_aug in zip(data_loader,data_loader_aug):
            
            #current_iter = i + 1
            current_iter += 1
            self._refresh_information(current_iter, lr=lr_scheduler.get_lr()[0])
            self.data_time.update(time.time() - end)

            # update feature extractor
            set_grad([self.model],True)
            set_grad([self.model_aug_bio],False)
            set_grad([self.model_aug_clo],False)
            self.model.train()
            self.model_aug_bio.eval()
            self.model_aug_clo.eval()
            isaug=True if current_iter>aug_start_iter else False
            loss,brightness_factors_p,contrast_factors_p,saturation_factors_p,brightness_factors_c,contrast_factors_c,saturation_factors_c = self.run(inputs, isaug=isaug)
            if current_iter%100==0:
                print(brightness_factors_p,contrast_factors_p,saturation_factors_p,brightness_factors_c,contrast_factors_c,saturation_factors_c)
            if self.this_task_info:
                loss = self.this_task_info.task_weight * loss
            optimizer.zero_grad()
            if self.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
                if self.this_task_info:
                    multitask_reduce_gradients(self.model, sync=True, ignore='classifier',
                                            task_grp=self.this_task_info.task_handle)
            optimizer.step()

            if current_iter>aug_start_iter:
                # update augmentor
                set_grad([self.model],False)
                set_grad([self.model_aug_bio],True)
                set_grad([self.model_aug_clo],True)
                self.model.eval()
                self.model_aug_bio.train()
                self.model_aug_clo.train()
                loss_aug,brightness_factors_p,contrast_factors_p,saturation_factors_p,brightness_factors_c,contrast_factors_c,saturation_factors_c = self.run(inputs_for_aug, isaug=True, forzen=True)
                loss_aug=-loss_aug
                if self.this_task_info:
                    loss_aug = self.this_task_info.task_weight * loss_aug
                optimizer_aug_bio.zero_grad()
                optimizer_aug_clo.zero_grad()
                if self.fp16:
                    with amp.scale_loss(loss_aug, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss_aug.backward()
                    
                    if self.this_task_info:
                        multitask_reduce_gradients(self.model_aug_bio, sync=True, ignore=None,
                                                task_grp=self.this_task_info.task_handle)
                        multitask_reduce_gradients(self.model_aug_clo, sync=True, ignore=None,
                                                task_grp=self.this_task_info.task_handle)
                optimizer_aug_bio.step()
                optimizer_aug_clo.step()

            self.batch_time.update(time.time() - end)
            end = time.time()

            self._logging(current_iter)
            local_rank = self.this_task_info.task_rank if self.this_task_info else dist.get_rank()
            if current_iter % self.args.save_freq == 0 and local_rank == 0:
                if test_loader is not None:
                    mAP = self._do_valid(test_loader, query, gallery, self.args.validate_feat)
                    if best_mAP < mAP:
                        best_mAP = mAP
                        best_iter = current_iter
                    end = time.time()

                save_checkpoint({'state_dict': self.model.state_dict()},
                                fpath=osp.join(self.args.logs_dir,
                                               'checkpoints',
                                               'checkpoint_{}.pth.tar'.format(current_iter)))

                print('\n * Finished iterations {:3d}. Best iter {:3d}, Best mAP {:4.1%}.\n'
                      .format(current_iter, best_iter, best_mAP))

            lr_scheduler.step()
            lr_scheduler_bio.step()
            lr_scheduler_clo.step()

    def _blend_batch(self, img1, img2, ratio):
        bound = 1.0 if img1.is_floating_point() else 255.0
        res = ratio * img1 + (1.0 - ratio) * img2
        return F.relu(res)-F.relu(res-bound)
        #return ().clamp(0, bound).to(img1.dtype)

    def _blend(self, img1, img2, ratio):
        bound = 1.0 if img1.is_floating_point() else 255.0
        res = ratio * img1 + (1.0 - ratio) * img2
        return F.relu(res)-F.relu(res-bound)
        #return ().clamp(0, bound).to(img1.dtype)


    def adjust_brightness(self, img, brightness_factor):
        """PRIVATE METHOD. Adjust brightness of an RGB image.

        .. warning::

            Module ``transforms.functional_tensor`` is private and should not be used in user application.
            Please, consider instead using methods from `transforms.functional` module.

        Args:
            img (Tensor): Image to be adjusted.
            brightness_factor (float):  How much to adjust the brightness. Can be
                any non negative number. 0 gives a black image, 1 gives the
                original image while 2 increases the brightness by a factor of 2.

        Returns:
            Tensor: Brightness adjusted image.
        """
        if brightness_factor < 0:
            raise ValueError('brightness_factor ({}) is not non-negative.'.format(brightness_factor))
        return self._blend(img, torch.zeros_like(img), brightness_factor)

    def adjust_brightness_batch(self, img, brightness_factor):
        """PRIVATE METHOD. Adjust brightness of an RGB image.

        .. warning::

            Module ``transforms.functional_tensor`` is private and should not be used in user application.
            Please, consider instead using methods from `transforms.functional` module.

        Args:
            img (Tensor): Image to be adjusted.
            brightness_factor (float):  How much to adjust the brightness. Can be
                any non negative number. 0 gives a black image, 1 gives the
                original image while 2 increases the brightness by a factor of 2.

        Returns:
            Tensor: Brightness adjusted image.
        """
        # if brightness_factor < 0:
        #     raise ValueError('brightness_factor ({}) is not non-negative.'.format(brightness_factor))
        return self._blend_batch(img, torch.zeros_like(img), brightness_factor)

    def rgb_to_grayscale(self, img):
        l_img = (0.2989 * img[:,0,:,:] + 0.587 * img[:,1,:,:] + 0.114 * img[:,2,:,:]).to(img.dtype)
        if l_img.ndim<4:
            l_img = l_img.unsqueeze(dim=-3)
        return l_img

    def adjust_contrast_batch(self, img, contrast_factor):
        """PRIVATE METHOD. Adjust contrast of an RGB image.

        .. warning::

            Module ``transforms.functional_tensor`` is private and should not be used in user application.
            Please, consider instead using methods from `transforms.functional` module.

        Args:
            img (Tensor): Image to be adjusted.
            contrast_factor (float): How much to adjust the contrast. Can be any
                non negative number. 0 gives a solid gray image, 1 gives the
                original image while 2 increases the contrast by a factor of 2.

        Returns:
            Tensor: Contrast adjusted image.
        """
        # if contrast_factor < 0:
        #     raise ValueError('contrast_factor ({}) is not non-negative.'.format(contrast_factor))

        dtype = img.dtype if torch.is_floating_point(img) else torch.float32
        mean = torch.mean(self.rgb_to_grayscale(img).to(dtype), dim=(-3, -2, -1), keepdim=True)

        return self._blend_batch(img, mean, contrast_factor)

    def adjust_contrast(self, img, contrast_factor):
        """PRIVATE METHOD. Adjust contrast of an RGB image.

        .. warning::

            Module ``transforms.functional_tensor`` is private and should not be used in user application.
            Please, consider instead using methods from `transforms.functional` module.

        Args:
            img (Tensor): Image to be adjusted.
            contrast_factor (float): How much to adjust the contrast. Can be any
                non negative number. 0 gives a solid gray image, 1 gives the
                original image while 2 increases the contrast by a factor of 2.

        Returns:
            Tensor: Contrast adjusted image.
        """
        if contrast_factor < 0:
            raise ValueError('contrast_factor ({}) is not non-negative.'.format(contrast_factor))

        dtype = img.dtype if torch.is_floating_point(img) else torch.float32
        mean = torch.mean(self.rgb_to_grayscale(img).to(dtype), dim=(-3, -2, -1), keepdim=True)

        return self._blend(img, mean, contrast_factor)

    def adjust_saturation_batch(self, img, saturation_factor):
        """PRIVATE METHOD. Adjust color saturation of an RGB image.

        .. warning::

            Module ``transforms.functional_tensor`` is private and should not be used in user application.
            Please, consider instead using methods from `transforms.functional` module.

        Args:
            img (Tensor): Image to be adjusted.
            saturation_factor (float):  How much to adjust the saturation. Can be any
                non negative number. 0 gives a black and white image, 1 gives the
                original image while 2 enhances the saturation by a factor of 2.

        Returns:
            Tensor: Saturation adjusted image.
        """
        # if saturation_factor < 0:
        #     raise ValueError('saturation_factor ({}) is not non-negative.'.format(saturation_factor))

        return self._blend_batch(img, self.rgb_to_grayscale(img), saturation_factor)

    def adjust_saturation(self, img, saturation_factor):
        """PRIVATE METHOD. Adjust color saturation of an RGB image.

        .. warning::

            Module ``transforms.functional_tensor`` is private and should not be used in user application.
            Please, consider instead using methods from `transforms.functional` module.

        Args:
            img (Tensor): Image to be adjusted.
            saturation_factor (float):  How much to adjust the saturation. Can be any
                non negative number. 0 gives a black and white image, 1 gives the
                original image while 2 enhances the saturation by a factor of 2.

        Returns:
            Tensor: Saturation adjusted image.
        """
        if saturation_factor < 0:
            raise ValueError('saturation_factor ({}) is not non-negative.'.format(saturation_factor))

        return self._blend(img, self.rgb_to_grayscale(img), saturation_factor)

    def do_augmentation(self, inputs, type='bio'):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        mean_tensor = torch.as_tensor(mean, dtype=inputs.dtype, device=inputs.device).view(1,-1,1,1)
        std_tensor = torch.as_tensor(std, dtype=inputs.dtype, device=inputs.device).view(1,-1,1,1)

        B = inputs.shape[0]
        model_aug = self.model_aug_bio if type=='bio' else self.model_aug_clo
        aug_factors = model_aug(inputs)
        
        brightness_factors, contrast_factors, saturation_factors = aug_factors[:,0].clone(), aug_factors[:,1].clone(), aug_factors[:,2].clone()
        
        brightness_factors_ = brightness_factors+torch.rand(1,device=brightness_factors.device)
        brightness_factors_ = brightness_factors_ *0.5+0.5
        contrast_factors_ = contrast_factors+torch.rand(1,device=contrast_factors.device)
        contrast_factors_ = contrast_factors_*0.5+0.5
        saturation_factors_ = saturation_factors+torch.rand(1,device=saturation_factors.device)
        saturation_factors_ = saturation_factors_*0.5+0.5
        assert len(brightness_factors_) ==len(contrast_factors_)==len(saturation_factors_)==B

        
        brightness_factors_ = brightness_factors_.reshape((-1,1,1,1))
        contrast_factors_ = contrast_factors_.reshape((-1,1,1,1))
        saturation_factors_ = saturation_factors_.reshape((-1,1,1,1))
        
        outputs = self.adjust_brightness_batch(inputs, brightness_factors_)
        outputs = self.adjust_contrast_batch(outputs, contrast_factors_)
        outputs = self.adjust_saturation_batch(outputs, saturation_factors_)
        outputs = (outputs-mean_tensor)/std_tensor

        return outputs,brightness_factors_[0],contrast_factors_[0],saturation_factors_[0]

    def run(self, inputs, isaug=False, forzen=False):
        inputs, clothes, targets = self._parse_data(inputs)
        brightness_factors_p,contrast_factors_p,saturation_factors_p,brightness_factors_c,contrast_factors_c,saturation_factors_c=None,None,None,None,None,None
        
        if isaug:
            inputs,brightness_factors_p,contrast_factors_p,saturation_factors_p = self.do_augmentation(inputs)
            clothes,brightness_factors_c,contrast_factors_c,saturation_factors_c = self.do_augmentation(clothes)
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


        logits, feat, logits_fusion, feat_fusion = self.model(inputs, clothes, forzen=forzen)

        if isinstance(logits_fusion, list):
            logits_global_fusion, logits_local_fusion = logits_fusion[0], logits_fusion[1:]
            
            if 'ce' in self.loss_option:
                loss_ce_global = self.ce_loss(logits_global_fusion.clone(), targets)
                loss_ce_local = [self.ce_loss(logit.clone(), targets) for logit in logits_local_fusion]
                
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
                
                loss_ce_global_bme = self.ce_loss(logits_global_bme.clone(), targets)
                loss_ce_local_bme = [self.ce_loss(logit.clone(), targets) for logit in logits_local_bme]
                
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
                loss_tr_global_fusion, _ = self.triplet_loss(feat_global_fusion.clone(), targets)
                loss_tr_local_fusion = [self.triplet_loss(feat_.clone(), targets)[0] for feat_ in feat_local_fusion]
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

        return loss,brightness_factors_p,contrast_factors_p,saturation_factors_p,brightness_factors_c,contrast_factors_c,saturation_factors_c
