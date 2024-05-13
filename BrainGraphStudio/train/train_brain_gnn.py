import torch
import torch.nn.functional as F
import numpy as np
import time
import logging
import nni
import os
from typing import Optional
from sklearn import metrics
from torch_geometric.loader import DataLoader
from torch.optim import lr_scheduler
from sklearn.model_selection import StratifiedKFold, train_test_split
from tensorboardX import SummaryWriter
from BrainGraphStudio.train.input_utils import ParamArgs
from BrainGraphStudio.train.train_utils import seed_everything, get_device
from BrainGraphStudio.models.model import build_model
from BrainGraphStudio.models.brainGNN.loss import topk_loss, consist_loss
import json
import sys

from BrainGraphStudio.utils import MaskableList
logger = logging.getLogger(__name__)

def train_and_evaluate(model, train_loader, test_loader, optimizer, scheduler, device, args, writer):
    model.train()
    accs, aucs, macros = [], [], []
    epoch_num = args.epochs

    for epoch in range(epoch_num):
        since  = time.time()
        tr_loss, s1_arr, s2_arr, w1, w2 = train(epoch, scheduler, optimizer, train_loader, model, device, args, writer) # collect accuracies
        val_loss = test_loss(test_loader, model, device, args)
        tr_acc = test_acc(train_loader, model, device)
        val_acc = test_acc(test_loader, model, device)
        
        train_micro, train_auc, train_macro = evaluate(model, device, train_loader)
        time_elapsed = time.time() - since

        logger.info('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        logger.info(f'(Train) | Epoch={epoch:03d}, loss={tr_loss:.4f}, '
                     f'train_micro={(train_micro * 100):.2f}, train_macro={(train_macro * 100):.2f}, '
                     f'train_auc={(train_auc * 100):.2f}')
        logger.info('Train Acc: {:.7f}, Test Loss: {:.7f}, Test Acc: {:.7f}'.format(tr_acc, val_loss, val_acc))

        if (epoch + 1) % args.test_interval == 0:
            test_micro, test_auc, test_macro = evaluate(model, device, test_loader)
            accs.append(test_micro)
            aucs.append(test_auc)
            macros.append(test_macro)
            text = f'(Train Epoch {epoch}), test_micro={(test_micro * 100):.2f}, ' \
                   f'test_macro={(test_macro * 100):.2f}, test_auc={(test_auc * 100):.2f}\n'
            logger.info(text)
        
        if args.use_nni:
            nni.report_intermediate_result(train_auc)
                                                        

        writer.add_scalars('Acc',{'train_acc':tr_acc,'val_acc':val_acc},  epoch)
        writer.add_scalars('Loss', {'train_loss': tr_loss, 'val_loss': val_loss},  epoch)
        writer.add_histogram('Hist/hist_s1', s1_arr, epoch)
        writer.add_histogram('Hist/hist_s2', s2_arr, epoch)
    
    accs, aucs, macros = np.sort(np.array(accs)), np.sort(np.array(aucs)), np.sort(np.array(macros))
    return accs.mean(), aucs.mean(), macros.mean()


def test_acc(loader, model, device):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        outputs= model(data.x, data.edge_index, data.batch, data.edge_attr,data.pos)
        pred = outputs[0].max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()

    return correct / len(loader.dataset)

@torch.no_grad()
def evaluate(model, device, loader, test_loader: Optional[DataLoader] = None) -> tuple[float, float]:
    model.eval()
    preds, trues, preds_prob = [], [], []

    correct, auc = 0, 0
    for data in loader:
        data = data.to(device)
        c, w1, w2, s1, s2 = model(data.x, data.edge_index, data.batch, data.edge_attr, data.pos)

        pred = c.max(dim=1)[1]
        preds += pred.detach().cpu().tolist()
        preds_prob += torch.exp(c)[:, 1].detach().cpu().tolist()
        trues += data.y.detach().cpu().tolist()

    try:
        train_auc = metrics.roc_auc_score(trues, preds_prob)
    except ValueError:
        logger.warning("Only one class present in y_true. ROC AUC score is not defined in that case.")
        train_auc = 0

    if np.isnan(auc):
        train_auc = 0.5
    train_micro = metrics.f1_score(trues, preds, average='micro')
    train_macro = metrics.f1_score(trues, preds, average='macro', labels=[0, 1])

    if test_loader is not None:
        test_micro, test_auc, test_macro = evaluate(model, device, test_loader)
        return train_micro, train_auc, train_macro, test_micro, test_auc, test_macro
    else:
        return train_micro, train_auc, train_macro


def test_loss(loader,model, device, args):
    logger.info('testing...........')
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
        for c in range(args.num_classes):
            loss_consist += consist_loss(s1[data.y == c], device)
        loss = args.lamb0*loss_c + args.lamb1 * loss_p1 + args.lamb2 * loss_p2 \
                   + args.lamb3 * loss_tpk1 + args.lamb4 *loss_tpk2 + args.lamb5* loss_consist

        loss_all += loss.item() * data.num_graphs
    return loss_all / len(loader.dataset)


def train(epoch, scheduler, optimizer, train_loader, model, device, args, writer):
    scheduler.step()

    # for param_group in optimizer.param_groups:
    #     print("LR", param_group['lr'])
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
            loss_consist += consist_loss(s1[data.y == c], device)
        loss = args.lamb0*loss_c + args.lamb1 * loss_p1 + args.lamb2 * loss_p2 \
                   + args.lamb3 * loss_tpk1 + args.lamb4 *loss_tpk2 + args.lamb5* loss_consist
        
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


def main_training_loop(path):
    best = None
    args = ParamArgs(path)
    logger.info("LOADED PARAM ARGS")
    if args.random_seed:
        seed_everything(args.random_seed)
        logger.info(f"Seeding All Random Processes with seed: {args.random_seed}")
    device = get_device()

    if args.use_nni:
        args.add_nni_args(nni.get_next_parameter())

    dataset, y = MaskableList(args.data_train_val), MaskableList(args.y_train_val)
    accs, aucs, macros = [], [], []

    if args.k_fold_splits < 2:
        logger.info("k fold splits < 2. No StratifiedKFolds in use")
        train_index, val_index = train_test_split(range(len(dataset)), test_size=0.2, stratify=y)
        # Proceed with the training using train_index and val_index as if they were from a single split
        indices = [(train_index, val_index)]
    else:
        skf = StratifiedKFold(n_splits=args.k_fold_splits, shuffle=True)
        indices = skf.split(dataset, y)

    for fold_idx, (train_index, val_index) in enumerate(indices):
        if args.use_nni:
            log_dir = os.path.join(os.environ["NNI_OUTPUT_DIR"], 'tensorboard', f'fold_{fold_idx}')
        else:
            log_dir = os.path.join(path, f".log/{fold_idx}")
        writer = SummaryWriter(log_dir=log_dir)

        model = build_model(args, device, args.model_name, args.num_features, args.num_nodes,
                            args.n_GNN_layers, n_classes=args.num_classes)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_scheduler_stepsize, gamma=args.gamma)
        
        
        train_set, val_set = dataset[train_index], dataset[val_index]
        train_loader = DataLoader(train_set, batch_size=args.batchsize, shuffle=False)
        val_loader = DataLoader(val_set, batch_size=args.batchsize, shuffle=False)

        val_micro, val_auc, val_macro = train_and_evaluate(model, train_loader, val_loader,
                                                                optimizer, scheduler, device, args, writer)
        
        accs.append(val_micro)
        aucs.append(val_auc)
        macros.append(val_macro)
        writer.close()

    result_str = f'(K Fold Final Result)| avg_acc={(np.mean(accs) * 100):.2f} +- {(np.std(accs) * 100): .2f}, ' \
                 f'avg_auc={(np.mean(aucs) * 100):.2f} +- {np.std(aucs) * 100:.2f}, ' \
                 f'avg_macro={(np.mean(macros) * 100):.2f} +- {np.std(macros) * 100:.2f}\n'
    logging.info(result_str)

    current_metric = np.mean(aucs)

    if args.use_nni:
        nni.report_final_result(current_metric)
    
    if best is None or current_metric > best:
        best = current_metric
        torch.save(model.state_dict(), os.path.join(args.path, 'best_model.pth')) 
        with open(os.path.join(args.path, "best_hyperparams.json"), "w") as hp_file:
            json.dump(args.nni_params, hp_file)


if __name__ == "__main__":
    logger.info('----------RUNNING TRAIN BRAINGNN SCRIPT-----------')
    assert(len(sys.argv) == 2)
    path = sys.argv[1]
    assert(os.path.exists(path))
    logger.info(f"DATA PATH: {path}")
    main_training_loop(path)