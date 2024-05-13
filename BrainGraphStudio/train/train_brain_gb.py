from BrainGraphStudio.train.input_utils import ParamArgs
from BrainGraphStudio.train.train_utils import seed_everything, get_device
from BrainGraphStudio.models.model import build_model
import numpy as np
import nni
import torch
import torch.nn.functional as F
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from typing import Optional
from torch.utils.data import DataLoader
import logging
import time
import json
import os
logger = logging.getLogger(__name__)



def train_and_evaluate(model, train_loader, test_loader, optimizer, device, args):
    model.train()
    accs, aucs, macros = [], [], []
    epoch_num = args.epochs

    for i in range(epoch_num):
        loss_all = 0
        for data in train_loader:
            data = data.to(device)

            # if args.mixup:
            #     data, y_a, y_b, lam = mixup(data)
            optimizer.zero_grad()
            out = model(data)

            # if args.mixup:
            #     loss = mixup_criterion(F.nll_loss, out, y_a, y_b, lam)
            # else:
            loss = F.nll_loss(out, data.y)

            loss.backward()
            optimizer.step()

            loss_all += loss.item()
        epoch_loss = loss_all / len(train_loader.dataset)

        train_micro, train_auc, train_macro = evaluate(model, device, train_loader)
        logger.info(f'(Train) | Epoch={i:03d}, loss={epoch_loss:.4f}, '
                     f'train_micro={(train_micro * 100):.2f}, train_macro={(train_macro * 100):.2f}, '
                     f'train_auc={(train_auc * 100):.2f}')

        if (i + 1) % args.test_interval == 0:
            test_micro, test_auc, test_macro = evaluate(model, device, test_loader)
            accs.append(test_micro)
            aucs.append(test_auc)
            macros.append(test_macro)
            text = f'(Train Epoch {i}), test_micro={(test_micro * 100):.2f}, ' \
                   f'test_macro={(test_macro * 100):.2f}, test_auc={(test_auc * 100):.2f}\n'
            logging.info(text)

        if args.enable_nni:
            nni.report_intermediate_result(train_auc)

    accs, aucs, macros = np.sort(np.array(accs)), np.sort(np.array(aucs)), np.sort(np.array(macros))
    return accs.mean(), aucs.mean(), macros.mean()


@torch.no_grad()
def evaluate(model, device, loader, test_loader: Optional[DataLoader] = None) -> tuple[float, float]:
    model.eval()
    preds, trues, preds_prob = [], [], []

    correct, auc = 0, 0
    for data in loader:
        data = data.to(device)
        c = model(data)

        pred = c.max(dim=1)[1]
        preds += pred.detach().cpu().tolist()
        preds_prob += torch.exp(c)[:, 1].detach().cpu().tolist()
        trues += data.y.detach().cpu().tolist()

    train_auc = metrics.roc_auc_score(trues, preds_prob)

    if np.isnan(auc):
        train_auc = 0.5
    train_micro = metrics.f1_score(trues, preds, average='micro')
    train_macro = metrics.f1_score(trues, preds, average='macro', labels=[0, 1])

    if test_loader is not None:
        test_micro, test_auc, test_macro = evaluate(model, device, test_loader)
        return train_micro, train_auc, train_macro, test_micro, test_auc, test_macro
    else:
        return train_micro, train_auc, train_macro


def main_training_loop(path):
    args = ParamArgs(path)
    seed_everything(args.seed)
    device = get_device()

    if args.use_nni:
        args.add_nni_args(nni.get_next_parameter())

    dataset, y = args.data_train_val, args.y_train_val
    accs, aucs, macros = [], [], []
    skf = StratifiedKFold(n_splits=args.k_fold_splits, shuffle=True)
    for train_index, val_index in skf.split(dataset, y):
        model = build_model(args, device, args.model_name, args.num_features, args.num_nodes,
                            args.n_MLP_layers, args.hidden_dim, args.num_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        train_set, val_set = dataset[train_index], dataset[val_index]

        train_loader = DataLoader(train_set, batch_size=args.train_batch_size, shuffle=False)
        val_loader = DataLoader(val_set, batch_size=args.test_batch_size, shuffle=False)

        # train
        val_micro, val_auc, val_macro = train_and_evaluate(model, train_loader, val_loader,
                                                                optimizer, device, args)

        val_micro, val_auc, val_macro = evaluate(model, device, val_loader)
        logging.info(f'(Initial Performance Last Epoch) | test_micro={(val_micro * 100):.2f}, '
                        f'test_macro={(val_macro * 100):.2f}, test_auc={(val_auc * 100):.2f}')

        accs.append(val_micro)
        aucs.append(val_auc)
        macros.append(val_macro)

    result_str = f'(K Fold Final Result)| avg_acc={(np.mean(accs) * 100):.2f} +- {(np.std(accs) * 100): .2f}, ' \
                 f'avg_auc={(np.mean(aucs) * 100):.2f} +- {np.std(aucs) * 100:.2f}, ' \
                 f'avg_macro={(np.mean(macros) * 100):.2f} +- {np.std(macros) * 100:.2f}\n'
    logging.info(result_str)

    current_metric = np.mean(aucs)

    if args.enable_nni:
        nni.report_final_result(current_metric)
    
    if nni.get_best_result() is None or current_metric > nni.get_best_result():
        torch.save(model.state_dict(), os.path.join(args.path, 'best_model.pth')) 
        with open(os.path.join(args.path, "best_hyperparams.json"), "w") as hp_file:
            json.dump(args.nni_params, hp_file)



    




