import timeit
import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from metrics import *
from net import Selfassembly
from GTDataset import GTDataset
import matplotlib.pyplot as plt
import matplotlib
import timeit
import dgl
import matplotlib
import pandas as pd

matplotlib.use('Agg')

device = torch.device('cuda')
print(f'use {device}')

def train(model, device, train_loader, optimizer):
    model.train()
    epoch_loss = 0.0
    for batch_idx, data in enumerate(train_loader):
        label = data[-1].to(device)
        protein_graph, = data[:-1]
        protein_graph = protein_graph.to(device)
        output = model(protein_graph)
        loss = criterion(output, label.view(-1, 1).float().to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()


    train_loss = epoch_loss / len(train_loader)

    return train_loss

def test(model, device, test_loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    with torch.no_grad():
        for data in test_loader:
            label = data[-1].to(device)
            protein_graph, = data[:-1]
            protein_graph = protein_graph.to(device)
            output = model(protein_graph)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, label.view(-1, 1).cpu()), 0)

    total_labels = total_labels.numpy().flatten()
    total_preds = total_preds.numpy().flatten()


    MSE = mse(total_labels, total_preds)
    MAE=mae(total_labels,total_preds)
    R2 = r2(total_labels,total_preds)
    return MSE,MAE,R2





if __name__ == '__main__':

    dataset = ''
    dataset_pre = ''
    file_path = ''

    fold = 1
    epochs = 1500
    batch = 1024
    lr = 1e-4


    train_set = GTDataset(dataset=dataset,
                          dataset_pre=dataset_pre,
                           protein_graph=file_path + '/train/protein_graph.bin',
                           protein_id=file_path + '/train/protein_id.npy',
                           label=file_path + '/train/label.npy')
    test_set = GTDataset(dataset=dataset,
                         dataset_pre=dataset_pre,
                           protein_graph=file_path + '/test/protein_graph.bin',
                           protein_id=file_path + '/test/protein_id.npy',
                           label=file_path + '/test/label.npy')

    train_loader = DataLoader(train_set, batch_size=batch, shuffle=True, collate_fn=train_set.collate, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch, shuffle=False, collate_fn=test_set.collate, drop_last=True)

    model = Selfassembly()
    device = torch.device('cuda')
    print(f'use {device}')
    model.to(device)

    start = timeit.default_timer()
    best_ci = 0
    best_mse = 100
    best_rm2 = 0
    best_epoch = -1
    best_mae = 1
    best_r2 = 0
    file_model = ''

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.9, patience=150,
                                                     verbose=True, min_lr=1e-6)

    Indexes = ('Epoch\t\tTime\t\tMSE\\t\tMAE\t\tR2\t\ttrainloss')


    train_losses = []
    test_mses = []
    print(Indexes)


    with open('test_mses.txt', 'a') as f_mse, open('train_losses.txt', 'a') as f_loss:
        for epoch in range(epochs):
            train_loss = train(model, device, train_loader, optimizer)
            mse_test, mae_test,R2_test= test(model, device, test_loader)
            scheduler.step(mse_test)
            end = timeit.default_timer()
            time = end - start

            ret = [epoch + 1, round(time, 2), round(mse_test, 6), 
                   round(mae_test, 7),round(R2_test,6),round(train_loss, 7)]
            print('\t\t'.join(map(str, ret)))

            train_losses.append(train_loss)
            test_mses.append(mse_test)

            f_mse.write(f'{mse_test:.6f}\n')
            f_loss.write(f'{train_loss:.6f}\n')

            if mse_test < best_mse:
                if mse_test < 0.600:
                    torch.save(model.state_dict(), file_model + 'Epoch:' + str(epoch + 1) +'state'+ '.pt')
                    torch.save(model,file_model + 'Epoch:' + str(epoch + 1) +'whole'+ '.pt')
                    print("model has been saved")
                best_epoch = epoch + 1
                best_mse = mse_test
                print('MSE improved at epoch ', best_epoch, ';\tbest_mse:', best_mse)





