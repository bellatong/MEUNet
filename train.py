#!/usr/bin/python3
# coding=utf-8

import sys
import datetime

"""sys.path.insert(0, '../')
sys.dont_write_bytecode = True"""

import dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from apex import amp
from MEUNet import MEUNet

def iou_loss(pred, mask):
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    iou = 1 - (inter + 1) / (union - inter + 1)
    return iou.mean()


def muti_iou_loss_fusion(d_united, d1, d2, d3, d4, labels_v):
    loss_united = iou_loss(d_united, labels_v)
    loss1 = iou_loss(d1, labels_v)
    loss2 = iou_loss(d2, labels_v)
    loss3 = iou_loss(d3, labels_v)
    loss4 = iou_loss(d4, labels_v)
    # loss5 = iou_loss(d5, labels_v)

    loss = loss_united + loss1 / 2 + loss2 / 4 + loss3 / 8 + loss4 / 16

    print("iou_l_united:%3f, iou_l1: %3f, iou_l2: %3f, iou_l3: %3f, iou_l4: %3f, iou_l: %3f" % (
        loss_united.item(), loss1.item(), loss2.item(), loss3.item(), loss4.item(), loss.item()))

    return loss, loss_united

# bce_loss = nn.BCELoss(size_average=True)

# bce_loss = nn.BCEWithLogitsLoss(size_average=True)

def muti_bce_loss_fusion(d_united, d1, d2, d3, d4, labels_v):
    loss_united = F.binary_cross_entropy_with_logits(d_united, labels_v)
    loss1 = F.binary_cross_entropy_with_logits(d1, labels_v)
    loss2 = F.binary_cross_entropy_with_logits(d2, labels_v)
    loss3 = F.binary_cross_entropy_with_logits(d3, labels_v)
    loss4 = F.binary_cross_entropy_with_logits(d4, labels_v)

    loss = loss_united + loss1 / 2 + loss2 / 4 + loss3 / 8 + loss4 / 16
    print("bce_l_united:%3f, bce_l1: %3f, bce_l2: %3f, bce_l3: %3f, bce_l4: %3f, bce_l: %3f" % (
        loss_united.item(), loss1.item(), loss2.item(), loss3.item(), loss4.item(), loss.item()))

    return loss, loss_united


def bce2d_new(input, target):
    assert (input.size() == target.size())
    pos = torch.eq(target, 1).float()
    neg = torch.eq(target, 0).float()
    # ing = ((torch.gt(target, 0) & torch.lt(target, 1))).float()

    num_pos = torch.sum(pos)
    num_neg = torch.sum(neg)
    num_total = num_pos + num_neg

    alpha = num_neg / num_total
    beta = 1.1 * num_pos / num_total
    # target pixel = 1 -> weight beta
    # target pixel = 0 -> weight 1-beta
    weights = alpha * pos + beta * neg

    loss = F.binary_cross_entropy_with_logits(input, target, weights, reduction='mean')

    print("edge_loss:%3f\n" % loss.item())

    return loss


def train(Dataset, Network):
    ## dataset
    cfg = Dataset.Config(datapath='/home/bianyetong/datasets/DUTS/DUTS-TR', savepath='./out', mode='train', batch=16,
                         lr=0.005, momen=0.9,decay=5e-4, epoch=600)
    data = Dataset.Data(cfg)
    loader = DataLoader(data, collate_fn=data.collate, batch_size=cfg.batch, shuffle=True, pin_memory=True,
                        num_workers=8)
    save_frq = 2
    ## network
    net = Network(cfg)
    net.train(True)
    net.cuda()
    ## parameter
    base, head = [], []
    for name, param in net.named_parameters():
        if 'bkbone.conv1' in name or 'bkbone.bn1' in name:
            print(name)
        elif 'bkbone' in name:
            base.append(param)
        else:
            head.append(param)
    optimizer = torch.optim.SGD([{'params': base}, {'params': head}], lr=cfg.lr, momentum=cfg.momen,
                                weight_decay=cfg.decay, nesterov=True)
    net, optimizer = amp.initialize(net, optimizer, opt_level='O1')
    sw = SummaryWriter(cfg.savepath)

    global_step = 0

    train_num = 10553

    loss_list = []
    iter_list = []

    for epoch in range(cfg.epoch):
        optimizer.param_groups[0]['lr'] = (1 - abs((epoch + 1) / (cfg.epoch + 1) * 2 - 1)) * cfg.lr * 0.1
        optimizer.param_groups[1]['lr'] = (1 - abs((epoch + 1) / (cfg.epoch + 1) * 2 - 1)) * cfg.lr

        for step, (image, mask, edge) in enumerate(loader):
            image, mask, edge = image.cuda(), mask.cuda(), edge.cuda()
            d_edge, d_united, d1, d2, d3, d4 = net(image)

            loss_bce, loss_bce_united = muti_bce_loss_fusion(d_united, d1, d2, d3, d4, mask)
            loss_iou, loss_iou_united = muti_iou_loss_fusion(d_united, d1, d2, d3, d4, mask)
            loss_edge = bce2d_new(d_edge, edge)
            tar_loss = loss_iou_united + loss_bce_united
            loss = 0.5 * loss_bce + 0.5 * loss_bce_united + 0.5 * loss_iou + 0.5 * loss_iou_united + loss_edge

            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scale_loss:
                scale_loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            iter_list.append(step+1)

            ## log
            global_step += 1
            # sw.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)
            # sw.add_scalars('loss', {'loss_bce': loss_bce.item(), 'loss_bce_united': loss_bce_united.item(),
            #                         'loss_iou': loss_iou.item(), ' loss_iou_united': loss_iou_united.item(),
            #                         'loss_edge': loss_edge, 'loss': loss.item()},
            #                global_step=global_step)

            # if step % 10 == 0:
            #     print(
            #         '%s | step:%d/%d/%d | lr=%.6f | lossb1=%.6f | lossd1=%.6f | loss1=%.6f | lossb2=%.6f | lossd2=%.6f | loss2=%.6f'
            #         % (datetime.datetime.now(), global_step, epoch + 1, cfg.epoch, optimizer.param_groups[0]['lr'],
            #            lossb1.item(), lossd1.item(), loss1.item(), lossb2.item(), lossd2.item(), loss2.item()))

            print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d]    train loss: %3f    loss:%3f" % (
                epoch + 1, cfg.epoch, (step + 1) * cfg.batch, train_num, step + 1, loss.item(), tar_loss.item()))

        if epoch % save_frq == 0:
            save_path = cfg.savepath + "/MEUNet_epoch_%d_itr_%d_train_%3f.pth" % (
                epoch, global_step, loss)
            torch.save(net.state_dict(), save_path)

        if step % 3000 == 0:
            with open("loss_recoder.txt", "w") as f:
                f.write("\n---------------------------------------------------\n")
                for (i, lo) in zip(iter_list, loss_list):
                    str_in = "%d:%.6f" % (i, lo)
                    f.write(str_in + "\n")


if __name__ == '__main__':
    train(dataset, MEUNet)
