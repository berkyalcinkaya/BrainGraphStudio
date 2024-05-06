import torch
import torch.nn.functional as F
import numpy as np
import time
import logging
from tensorboardX import SummaryWriter
logger = logging.getLogger(__name__)

EPS = 1e-10

def train_and_evaluate(model, train_loader, test_loader, optimiizer, device, args):
    model.train()
    model.train()
    accs, aucs, macros = [], [], []
    epoch_num = args.epochs

    for epoch in range(epoch_num):
        since  = time.time()
        tr_loss, s1_arr, s2_arr, w1, w2 = train(epoch)
        tr_acc = test_acc(train_loader)
        val_acc = test_acc(test_loader)
        val_loss = test_loss(test_loader,epoch)
        time_elapsed = time.time() - since
        logger.info('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        logger.info('Epoch: {:03d}, Train Loss: {:.7f}, '
            'Train Acc: {:.7f}, Test Loss: {:.7f}, Test Acc: {:.7f}'.format(epoch, tr_loss,
                                                        tr_acc, val_loss, val_acc))

        writer.add_scalars('Acc',{'train_acc':tr_acc,'val_acc':val_acc},  epoch)
        writer.add_scalars('Loss', {'train_loss': tr_loss, 'val_loss': val_loss},  epoch)
        writer.add_histogram('Hist/hist_s1', s1_arr, epoch)
        writer.add_histogram('Hist/hist_s2', s2_arr, epoch)




def test_acc(loader, model, device):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        outputs= model(data.x, data.edge_index, data.batch, data.edge_attr,data.pos)
        pred = outputs[0].max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()

    return correct / len(loader.dataset)

def test_loss(loader,epoch, model, device, args):
    print('testing...........')
    model.eval()
    loss_all = 0
    for data in loader:
        data = data.to(device)
        output, w1, w2, s1, s2= model(data.x, data.edge_index, data.batch, data.edge_attr,data.pos)
        loss_c = F.nll_loss(output, data.y)

        loss_p1 = (torch.norm(w1, p=2)-1) ** 2
        loss_p2 = (torch.norm(w2, p=2)-1) ** 2
        loss_tpk1 = topk_loss(s1,args.pooling_ratio)
        loss_tpk2 = topk_loss(s2,args.pooling_ratio)
        loss_consist = 0
        for c in range(nclass):
            loss_consist += consist_loss(s1[data.y == c])
        loss = lamb0*loss_c + lamb1 * loss_p1 + lamb2 * loss_p2 \
                   + lamb3 * loss_tpk1 + lamb4 *loss_tpk2 + lamb5* loss_consist

        loss_all += loss.item() * data.num_graphs
    return loss_all / len(loader.dataset)

def topk_loss(s,ratio):
    if ratio > 0.5:
        ratio = 1-ratio
    s = s.sort(dim=1).values
    res =  -torch.log(s[:,-int(s.size(1)*ratio):]+EPS).mean() -torch.log(1-s[:,:int(s.size(1)*ratio)]+EPS).mean()
    return res


def consist_loss(s, device):
    if len(s) == 0:
        return 0
    s = torch.sigmoid(s)
    W = torch.ones(s.shape[0],s.shape[0])
    D = torch.eye(s.shape[0])*torch.sum(W,dim=1)
    L = D-W
    L = L.to(device)
    res = torch.trace(torch.transpose(s,0,1) @ L @ s)/(s.shape[0]*s.shape[0])
    return res


def train(epoch, scheduler, optimizer, train_loader, model, device, args):
    print('train...........')
    scheduler.step()

    for param_group in optimizer.param_groups:
        print("LR", param_group['lr'])
    model.train()
    s1_list = []
    s2_list = []
    loss_all = 0
    step = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        #print(data.x.shape, data.edge_index.shape, data.batch.shape, data.edge_attr.shape, data.pos.shape)
        output, w1, w2, s1, s2 = model(data.x, data.edge_index, data.batch, data.edge_attr, data.pos)
        s1_list.append(s1.view(-1).detach().cpu().numpy())
        s2_list.append(s2.view(-1).detach().cpu().numpy())

        loss_c = F.nll_loss(output, data.y)

        loss_p1 = (torch.norm(w1, p=2)-1) ** 2
        loss_p2 = (torch.norm(w2, p=2)-1) ** 2
        loss_tpk1 = topk_loss(s1,args.pooling_ratio)
        loss_tpk2 = topk_loss(s2,args.pooling_ratio)
        loss_consist = 0
        for c in range(args.num_classes):
            loss_consist += consist_loss(s1[data.y == c])
        loss = lamb0*loss_c + lamb1 * loss_p1 + lamb2 * loss_p2 \
                   + lamb3 * loss_tpk1 + lamb4 *loss_tpk2 + lamb5* loss_consist
        writer.add_scalar('train/classification_loss', loss_c, epoch*len(train_loader)+step)
        writer.add_scalar('train/unit_loss1', loss_p1, epoch*len(train_loader)+step)
        writer.add_scalar('train/unit_loss2', loss_p2, epoch*len(train_loader)+step)
        writer.add_scalar('train/TopK_loss1', loss_tpk1, epoch*len(train_loader)+step)
        writer.add_scalar('train/TopK_loss2', loss_tpk2, epoch*len(train_loader)+step)
        writer.add_scalar('train/GCL_loss', loss_consist, epoch*len(train_loader)+step)
        step = step + 1

        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()

        s1_arr = np.hstack(s1_list)
        s2_arr = np.hstack(s2_list)
    return loss_all / len(train_loader.dataset), s1_arr, s2_arr ,w1,w2