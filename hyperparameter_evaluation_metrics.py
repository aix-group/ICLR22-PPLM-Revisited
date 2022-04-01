# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 16:18:56 2022

"""

import math
from typing import List

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import (GPT2Tokenizer, OpenAIGPTLMHeadModel,
                          OpenAIGPTTokenizer)
from transformers.modeling_gpt2 import GPT2LMHeadModel

from dist import eval_distinct
from run_pplm import run_pplm_example
from run_pplm_simple_text import postprocess_pplm_output, pplm_loss

from IPython.display import display, HTML
import glob
import re

def clean_text(t):
    t = t.replace('<|endoftext|>', '')
    t = re.sub('\s{1,}', ' ', t)
    return t


def evaluate(samples):
    df = pd.DataFrame({
        'text': [clean_text(sample) for sample in samples]
    })
    df['ppl'] = df['text'].apply(compute_ppl)
    dist = df['text'].apply(lambda text: eval_distinct([text], PPLM_TOKENIZER))
    df[['dist-1', 'dist-2', 'dist-3']] = pd.DataFrame(dist.tolist())
    df['dist'] = dist.apply(np.mean)
    
    '''
    # Readability metrics
    df['flesch_reading_ease'] = df['text'].apply(textstat.flesch_reading_ease)
    df['gunning_fog'] = df['text'].apply(textstat.gunning_fog)
    df['automated_readability_index'] = df['text'].apply(textstat.automated_readability_index)
    df['coleman_liau_index'] = df['text'].apply(textstat.coleman_liau_index)
    df['lexicon_count'] = df['text'].apply(textstat.lexicon_count)
    df['syllable_count'] = df['text'].apply(textstat.syllable_count)
    df['sentence_count'] = df['text'].apply(textstat.sentence_count)
    df['precision_english_1k'] = df['text'].apply(precision_english_1k)
    df['precision_english_2k'] = df['text'].apply(precision_english_2k)
    df['precision_english_5k'] = df['text'].apply(precision_english_5k)
    '''
    return df

def compute_ppl(sent):
    indexed_tokens = ppl_tokenizer.encode(sent)
    tokens_tensor = torch.tensor([indexed_tokens])
    with torch.no_grad():
        outputs = ppl_model.forward(tokens_tensor, labels=tokens_tensor)
    loss = outputs[0]
    return math.exp(loss.item())

METRICS = ['ppl','dist']

ppl_tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
ppl_model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
_ = ppl_model.eval()

PPLM_TOKENIZER = GPT2Tokenizer.from_pretrained("gpt2-medium")

filenames_allprompts = glob.glob('hyperparameter-files/allprompts*.csv')


for i,file in enumerate(filenames_allprompts):
    print("index:",i)
    # Load raw data
    df = pd.read_csv(file, index_col=0)
    df = df[df.index.notnull()].reset_index(drop=True)
    
    # Compute metrics
    df = pd.concat([df, evaluate(df['raw'])], axis=1)
    
    # Split unperturbed/perturbed samples
    df = df[~(df['kind'] == 'unperturbed')]

    df['PPLM objective'] = file.split('/')[1].split('allprompts_')[1].strip('.csv')
    
    if i==0:
        df_combined=df
    else:
        df_combined = pd.concat([df_combined,df], ignore_index=True)
    
    df_agg = df_combined[METRICS + ['PPLM objective']].groupby('PPLM objective').agg(['mean', 'std']).round(2)
    #print(df_agg.head())
df_agg.to_csv('hyperparameter_evaluation_metrics.csv')
