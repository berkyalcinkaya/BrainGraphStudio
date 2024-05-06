import torch
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy



def train_model(model, name, train_loader, test_loader, criterion = torch.nn.BCELoss(), LR = 0.001, epochs = 300, patience = 20):
    
    def train(train_loader, model):
        model.train()

        for data in train_loader:  # Iterate in batches over the training dataset.
            out, embedding = model(data.x.float(), data.edge_index, data.edge_weight, data.batch)  # Perform a single forward pass.
            loss = criterion(out.float(), data.y.float())  # Compute the loss.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.
        return loss

    def test(loader, model):
        model.eval()

        correct = 0
        for data in loader:  # Iterate in batches over the training/test dataset.
            out, embedding = model(data.x.float(), data.edge_index, data.edge_weight, data.batch)  
            pred = (out>0.5).long()
            correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        return correct / len(loader.dataset)  # Derive ratio of correct predictions.
    
    def save_best_acc():
        with open(join(model_dir, "acc.txt"), "w") as f:
            f.write(str(best_acc))
        return
    
    print(model)
    print("Number of parameters: ", sum(p.numel() for p in model.parameters()))
    
    model_dir = make_model_dir(name)
    weight_path = os.path.join(model_dir, "weights.pth")
    hyperparams = {"loss": [str(criterion)],
                    "batch_size": [train_loader.batch_size],
                    "lr": [LR], 
                    "epochs": [epochs],
                    "model": [str(type(model))]}
    print(hyperparams)
    pd.DataFrame.from_dict(hyperparams).to_csv(os.path.join(model_dir,"params.csv" ))

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    improvement_counteer = 0 
    losses = []
    test_accs = []
    train_accs = []
    best_acc = 0.0   # init to negative infinity
    best_weights = None
    for epoch in tqdm(range(epochs)):
        loss = train(train_loader, model)
        losses.append(loss)
        train_acc = test(train_loader, model)
        test_acc = test(test_loader, model)
        test_accs.append(test_acc)
        train_accs.append(train_acc)
        
        if test_acc > best_acc:
            print("FOUND NEW BEST")
            best_acc = test_acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), weight_path)
            counter = 0
        else:
            counter +=1
        
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Best So Far: {best_acc:.4f}, Epochs Since Improvement: {counter}')
        
        if counter > patience:
            print("Early stopping triggered. Best validation accuracy: {:.4f}".format(best_acc))
            break


    make_loss_curve(losses, model_dir)
    make_test_acc_curve(test_accs, model_dir)
    save_best_acc()

def make_model_dir(model_name, model_location = "models"):
    model_dir = os.path.join(model_location, model_name)
    try:
        os.mkdir(model_dir)
    except FileExistsError:
        pass
    return model_dir

def make_loss_curve(losses, model_dir):
    losses_float = [float(loss.cpu().detach().numpy()) for loss in losses] 
    loss_indices = [i for i,l in enumerate(losses_float)] 
    plt.plot(loss_indices, losses_float)
    plt.title("Train Losses")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.savefig(os.path.join(model_dir, "loss.png"))
    plt.show()

def make_test_acc_curve(losses, model_dir):
    plt.plot(losses)
    plt.title("Test Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.savefig(os.path.join(model_dir, "test_acc.png"))
    plt.show()



