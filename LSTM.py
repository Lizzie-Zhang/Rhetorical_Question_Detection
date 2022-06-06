#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 08:46:24 2022

@author: lizziezhang
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):  
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim)     
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, embedded):
        embedded = embedded.unsqueeze(0)      
        output, (hidden, cell) = self.lstm(embedded)
        assert torch.equal(output[-1,:,:], hidden.squeeze(0))
        return self.fc(hidden.squeeze(0))
    
def train_epoch(model, train_loader, optimizer, criterion):
    model.train(True)
    total_loss = 0.0

    for inputs, labels in train_loader:
        inputs = torch.tensor(inputs)
        inputs = inputs.to(torch.float32)
        
        labels = labels.to(torch.float)
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(1), labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)

    epoch_loss = total_loss / len(train_loader.dataset)
    return epoch_loss

def eval_epoch(model, valid_loader, criterion):
    model.train(False)
    total_loss = 0.0
    
    for inputs, labels in valid_loader:
        inputs = torch.tensor(inputs)
        inputs = inputs.to(torch.float32)
        
        labels = labels.to(torch.float)
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(1), labels)
        loss.backward()
        
        total_loss += loss.item() * inputs.size(0)
        
    epoch_loss = total_loss / len(valid_loader.dataset)
    return epoch_loss

def train_model(model, model_name, train_loader, valid_loader, test_loader, optimizer, criterion, num_epochs=20):
    best_loss = np.inf
    results = {}
    results['train'], results['valid'], results['test'] = [], [], []
    for epoch in tqdm(range(num_epochs)):
        test_loss = eval_epoch(model, test_loader, criterion)
        results['test'].append(test_loss)
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        results['train'].append(train_loss)
        
        valid_loss = eval_epoch(model, valid_loader, criterion)
        results['valid'].append(valid_loss)
        
        if valid_loss < best_loss:
            best_loss = valid_loss
            results['best_model'] = model
            
        torch.save(results, model_name+'_results.pt')
        
        
        
data = [[torch.tensor(X[i]), 
         questions_with_label['label'].iloc[i]] for i in range(len(X))]

train, test = train_test_split(data, test_size=0.2, random_state=12345)
train, val = train_test_split(train, test_size=0.25, random_state=12345)

batch_size = 128
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(val, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=True)

model = LSTM(input_dim=768, hidden_dim=500, output_dim=1)
optimizer = optim.Adam(model.parameters(), lr=0.00001)
criterion = nn.MSELoss()
train_model(model, 'LSTM', train_loader, valid_loader, test_loader, optimizer, criterion, num_epochs=50)




results = torch.load('LSTM_results.pt')
best_lstm = results['best_model']
with torch.no_grad():
    preds = torch.empty(0)
    trues = torch.empty(0)
    for inputs, labels in test_loader:
            inputs = torch.tensor(inputs)
            inputs = inputs.to(torch.float32)

            labels = labels.to(torch.float)
            trues = torch.cat((trues, labels), 0)

            outputs = model(inputs)
            preds = torch.cat((preds, outputs), 0) 
            

    print(confusion_matrix(trues,preds>0.5))  
    print(classification_report(trues,preds>0.5))
    print("Accuracy: ")
    print(round(accuracy_score(trues,preds>0.5)*100,2),"%")






















