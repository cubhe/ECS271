# Created by Renzhi He, UCDavis, 2024

from NeRF import MLP

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

def train(model, dataloader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}')

def test(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, targets in dataloader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    print(f'Accuracy: {100 * correct / total}%')
def main():
    writer = SummaryWriter('runs/experiment_2')

    # File path to your Excel dataset
    file_path = 'data.xlsx'
    classes=10
    # Create the dataset
    # load data
    filepath = './studentsdigits-train.csv'
    data_train = pd.read_csv(filepath, header=None).values[1:, :]
    lens=data_train.shape[0]
    x = data_train[:, :8]
    x = np.array(x).astype('uint8')
    y_org = data_train[:, 8]
    y_org = np.array(y_org).astype('uint8')
    y=np.zeros((lens,classes))
    for i in range(lens):
        y[i,int(y_org[i])] = 1

    print(x.shape, y.shape)
    x=torch.from_numpy(x).float().cuda()
    y=torch.from_numpy(y).float().cuda()
    id = np.random.permutation(x.shape[0])
    x_t=x[id[:int(lens*0.85)]]
    y_t=y[id[:int(lens*0.85)]]
    x_v=x[id[int(lens*0.85):]]
    y_v=y[id[int(lens*0.85):]]

    # Initialize the model
    model = MLP()
    model.cuda()

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Train the model
    epochs = 1250  # You can adjust the number of epochs
    batch_size = 500
    global_step=0
    loss=0

    model.train()
    for epoch in range(epochs):
        id = np.random.permutation(x_t.shape[0])
        for i in range(int(x_t.shape[0]/batch_size)-1):
            global_step+=1

            traning_data    = x_t[id[batch_size * i:batch_size * (i+1)]]
            traning_label   = y_t[id[batch_size * i:batch_size * (i+1)]]


            optimizer.zero_grad()
            outputs = model(traning_data)
            loss = criterion(traning_label,outputs)
            # loss += loss.item()
            loss.backward()
            optimizer.step()

            writer.add_scalar('Training loss', loss.item(), global_step)
        print(f'Epoch {epoch + 1}, Steps {global_step}, Loss: {loss}')

    # Test the model
    # test(model, test_loader)

    writer.close()
# Entry point of the script
if __name__ == '__main__':
    main()
