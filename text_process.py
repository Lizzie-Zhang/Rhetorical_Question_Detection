#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 08:33:48 2022

@author: lizziezhang
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from transformers import BertTokenizer, BertModel


#load data
questions_with_label = pd.read_csv('text.csv')

#word embeddings
pretrained_lm = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(pretrained_lm, do_lower_case=True)
model = BertModel.from_pretrained('bert-base-uncased')

pd.DataFrame().to_csv('inputs.csv')
with torch.no_grad():
    for i in tqdm(X):
        tokenized = tokenizer(i, padding=False, return_tensors='pt')
        embedded = model(tokenized['input_ids']).last_hidden_state[0][-1].detach().numpy()
        inputs_df = pd.DataFrame([embedded])
        inputs_df.to_csv('inputs.csv', mode='a', index=False, header=False)


#If you want to use input.csv as your input, please use the following code         
#inputs = pd.read_csv('inputs.csv', sep='/n')
#X = np.array([np.array(inputs['""'][i].split(',')).astype(float) for i in range(len(inputs))])