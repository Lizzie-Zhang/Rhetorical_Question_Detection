#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 08:28:24 2022

@author: lizziezhang
"""

import pandas as pd
from tqdm import tqdm
import pysentiment2 as ps

#Load text data
df = pd.read_csv('text.csv')

#count question mark number
question_mark_num = []
for text in tqdm(df['text']):
    question_mark_num.append(text.count('?'))
df['question_mark_num'] = question_mark_num

#calculate sentiment polarity and subjectivity
senti_polarity = []
subjectivity = []

for text in tqdm(df['text']):
    lm = ps.LM()
    tokens = lm.tokenize(text)
    lm_score = lm.get_score(tokens)
    senti_polarity.append(lm_score['Polarity'])
    subjectivity.append(lm_score['Subjectivity'])

df['senti_polarity'] = senti_polarity
df['subjectivity'] = subjectivity

df.to_csv('feature.csv')
