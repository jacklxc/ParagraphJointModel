import torch
import jsonlines
import os

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup, AutoModelForSequenceClassification
from tqdm import tqdm
from typing import List
from sklearn.metrics import f1_score, precision_score, recall_score

import math
import numpy as np

from tqdm import tqdm
from util import read_passages, clean_words, test_f1, to_BIO, from_BIO


class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y
    
class TimeDistributedDense(nn.Module):
    def __init__(self, INPUT_SIZE, OUTPUT_SIZE):
        super(TimeDistributedDense, self).__init__()
        self.input_size = INPUT_SIZE
        self.output_size = OUTPUT_SIZE
        self.linear = nn.Linear(INPUT_SIZE, OUTPUT_SIZE, bias=True)
        self.timedistributedlayer = TimeDistributed(self.linear)
    def forward(self, x):
        # x: (BATCH_SIZE, ARRAY_LEN, INPUT_SIZE)
        
        return self.timedistributedlayer(x)
    
class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, hidden_size, num_labels, hidden_dropout_prob = 0.1):
        super().__init__()
        self.dense = TimeDistributedDense(hidden_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.out_proj = TimeDistributedDense(hidden_size, num_labels)

    def forward(self, x, **kwargs):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class WordAttention(nn.Module):
    """
    x: (BATCH_SIZE, N_sentence, N_token, INPUT_SIZE)
    token_mask: (batch_size, N_sep, N_token)
    out: (BATCH_SIZE, N_sentence, INPUT_SIZE)
    mask: (BATCH_SIZE, N_sentence)
    """
    def __init__(self, INPUT_SIZE, PROJ_SIZE, dropout = 0.1):
        super(WordAttention, self).__init__()
        self.activation = torch.tanh
        self.att_proj = TimeDistributedDense(INPUT_SIZE, PROJ_SIZE)
        self.dropout = nn.Dropout(dropout)
        self.att_scorer = TimeDistributedDense(PROJ_SIZE, 1)
        
    def forward(self, x, token_mask):
        proj_input = self.att_proj(self.dropout(x.view(-1, x.size(-1))))
        proj_input = self.dropout(self.activation(proj_input))
        raw_att_scores = self.att_scorer(proj_input).squeeze(-1).view(x.size(0),x.size(1),x.size(2)) # (Batch_size, N_sentence, N_token)
        att_scores = F.softmax(raw_att_scores.masked_fill((1 - token_mask).bool(), float('-inf')), dim=-1)
        att_scores = torch.where(torch.isnan(att_scores), torch.zeros_like(att_scores), att_scores) # Replace NaN with 0
        batch_att_scores = att_scores.view(-1, att_scores.size(-1)) # (Batch_size * N_sentence, N_token)
        out = torch.bmm(batch_att_scores.unsqueeze(1), x.view(-1, x.size(2), x.size(3))).squeeze(1) 
        # (Batch_size * N_sentence, INPUT_SIZE)
        out = out.view(x.size(0), x.size(1), x.size(-1))
        mask = token_mask[:,:,0]
        return out, mask

class DynamicSentenceAttention(nn.Module):
    """
    input: (BATCH_SIZE, N_sentence, INPUT_SIZE)
    output: (BATCH_SIZE, INPUT_SIZE)
    """
    def __init__(self, INPUT_SIZE, PROJ_SIZE, REC_HID_SIZE = None, dropout = 0.1):
        super(DynamicSentenceAttention, self).__init__()
        self.activation = torch.tanh
        self.att_proj = TimeDistributedDense(INPUT_SIZE, PROJ_SIZE)
        self.dropout = nn.Dropout(dropout)
        
        if REC_HID_SIZE is not None:
            self.contextualized = True
            self.lstm = nn.LSTM(PROJ_SIZE, REC_HID_SIZE, bidirectional = False, batch_first = True)
            self.att_scorer = TimeDistributedDense(REC_HID_SIZE, 2)
        else:
            self.contextualized = False
            self.att_scorer = TimeDistributedDense(PROJ_SIZE, 2)
        
    def forward(self, sentence_reps, sentence_mask, att_scores, valid_scores):
        # sentence_reps: (BATCH_SIZE, N_sentence, INPUT_SIZE)
        # sentence_mask: (BATCH_SIZE, N_sentence)
        # att_scores: (BATCH_SIZE, N_sentence)
        # valid_scores: (BATCH_SIZE, N_sentence)
        # result: (BATCH_SIZE, INPUT_SIZE)
        #att_scores = rationale_out[:,:,1] # (BATCH_SIZE, N_sentence)
        #valid_scores = rationale_out[:,:,1] > rationale_out[:,:,0] # Only consider sentences predicted as rationales
        sentence_mask = torch.logical_and(sentence_mask, valid_scores)
        
        # Force those sentence representations in paragraph without rationale to be 0. 
        #NEI_mask = (torch.sum(sentence_mask, axis=1) > 0).long().unsqueeze(-1).expand(-1, sentence_reps.size(-1))
        
        if sentence_reps.size(0) > 0:
            att_scores = F.softmax(att_scores.masked_fill((~sentence_mask).bool(), -1e4), dim=-1)
            #att_scores = torch.where(torch.isnan(att_scores), torch.zeros_like(att_scores), att_scores) # Replace NaN with 0
            result = torch.bmm(att_scores.unsqueeze(1), sentence_reps).squeeze(1)
            return result# * NEI_mask 
        else:
            return sentence_reps[:,0,:]# * NEI_mask
  
        
class JointParagraphClassifier(nn.Module):
    def __init__(self, bert_path, bert_dim, dropout = 0.1, ignore_index=2):
        super(JointParagraphClassifier, self).__init__()
        self.stance_label_size = 3
        self.rationale_label_size = 2
        self.ignore_index = 2
        self.bert = AutoModel.from_pretrained(bert_path)  
        self.stance_criterion = nn.CrossEntropyLoss()
        self.rationale_criterion = nn.CrossEntropyLoss(ignore_index = 2)
        self.dropout = dropout
        self.bert_dim = bert_dim
        self.sentence_attention = DynamicSentenceAttention(bert_dim, bert_dim, dropout=dropout)
        self.word_attention = WordAttention(bert_dim, bert_dim, dropout=dropout)
        self.rationale_linear = ClassificationHead(bert_dim, self.rationale_label_size, hidden_dropout_prob = dropout)
        self.stance_linear = ClassificationHead(bert_dim, self.stance_label_size, hidden_dropout_prob = dropout)
        self.extra_modules = [
            self.sentence_attention,
            self.word_attention,
            self.rationale_linear,
            self.stance_linear,
            self.stance_criterion,
            self.rationale_criterion
        ]
            
    def reinitialize(self):
        self.extra_modules = []
        self.rationale_linear = ClassificationHead(self.bert_dim, self.rationale_label_size, hidden_dropout_prob = self.dropout)
        self.stance_linear = ClassificationHead(self.bert_dim, self.stance_label_size, hidden_dropout_prob = self.dropout)
        self.sentence_attention = DynamicSentenceAttention(self.bert_dim, self.bert_dim, dropout=self.dropout)
        self.extra_modules = [
            self.rationale_linear,
            self.stance_linear,
            self.stance_criterion,
            self.rationale_criterion,
            self.word_attention,
            self.sentence_attention
        ]
    
    def forward(self, encoded_dict, transformation_indices, stance_label = None, rationale_label = None, sample_p=1, rationale_score = False):
        batch_indices, indices_by_batch, mask = transformation_indices # (batch_size, N_sep, N_token)
        bert_out = self.bert(**encoded_dict)[0] # (BATCH_SIZE, sequence_len, BERT_DIM)
        bert_tokens = bert_out[batch_indices, indices_by_batch, :]
        # bert_tokens: (batch_size, N_sep, N_token, BERT_dim)
        sentence_reps, sentence_mask = self.word_attention(bert_tokens, mask) 
        # (Batch_size, N_sep, BERT_DIM), (Batch_size, N_sep)
        #print(bert_out.shape, bert_tokens.shape, sentence_reps.shape, sentence_mask.shape, rationale_label.shape)
        rationale_out = self.rationale_linear(sentence_reps) # (Batch_size, N_sep, 2)
        att_scores = rationale_out[:,:,1] # (BATCH_SIZE, N_sentence)
        
        if bool(torch.rand(1) < sample_p): # Choose sentence according to predicted rationale
            valid_scores = rationale_out[:,:,1] > rationale_out[:,:,0]
        else:
            valid_scores = rationale_label == 1 # Ground truth
        paragraph_rep = self.sentence_attention(sentence_reps, sentence_mask, att_scores, valid_scores) 
        # (BATCH_SIZE, BERT_DIM) 
        
        stance_out = self.stance_linear(paragraph_rep) # (Batch_size, 3)
        
        if stance_label is not None:
            stance_loss = self.stance_criterion(stance_out, stance_label)
        else:
            stance_loss = None
        if rationale_label is not None:
            rationale_loss = self.rationale_criterion(rationale_out.view(-1, self.rationale_label_size), 
                                                      rationale_label.view(-1)) # ignore index 2
        else:
            rationale_loss = None
            
        stance_out = torch.argmax(stance_out.cpu(), dim=-1).detach().numpy().tolist()
        if rationale_score:
            rationale_pred = rationale_out.cpu()[:,:,1] # (Batch_size, N_sep)
        else:
            rationale_pred = torch.argmax(rationale_out.cpu(), dim=-1) # (Batch_size, N_sep)
        rationale_out = [rationale_pred_paragraph[mask].detach().numpy().tolist() for rationale_pred_paragraph, mask in zip(rationale_pred, sentence_mask.bool())]            
        return rationale_out, stance_out, rationale_loss, stance_loss
    
class DomainAdaptationJointParagraphClassifier(nn.Module):
    def __init__(self, bert_path, bert_dim, dropout = 0.1, ignore_index=2):
        super(DomainAdaptationJointParagraphClassifier, self).__init__()
        self.stance_label_size = 3
        self.rationale_label_size = 2
        self.ignore_index = 2
        self.bert = AutoModel.from_pretrained(bert_path)  
        self.stance_criterion = nn.CrossEntropyLoss()
        self.rationale_criterion = nn.CrossEntropyLoss(ignore_index = 2)
        self.rationale_linear_fever = ClassificationHead(bert_dim, self.rationale_label_size, hidden_dropout_prob = dropout)
        self.rationale_linear_scifact = ClassificationHead(bert_dim, self.rationale_label_size, hidden_dropout_prob = dropout)
        self.stance_linear_scifact = ClassificationHead(bert_dim, self.stance_label_size, hidden_dropout_prob = dropout)
        self.stance_linear_fever = ClassificationHead(bert_dim, self.stance_label_size, hidden_dropout_prob = dropout)

        self.sentence_attention_scifact = DynamicSentenceAttention(bert_dim, bert_dim, dropout=dropout)
        self.sentence_attention_fever = DynamicSentenceAttention(bert_dim, bert_dim, dropout=dropout)
        self.word_attention = WordAttention(bert_dim, bert_dim, dropout=dropout)
        
        self.extra_modules = [
            self.word_attention,
            self.sentence_attention_scifact,
            self.sentence_attention_fever,
            self.stance_linear_fever,
            self.stance_linear_scifact,
            self.rationale_linear_fever,
            self.rationale_linear_scifact,
            self.stance_criterion,
            self.rationale_criterion
        ]
        
    def forward(self, encoded_dict, transformation_indices, domain_indices, stance_label = None, rationale_label = None, sample_p=1):
        batch_indices, indices_by_batch, mask = transformation_indices # (batch_size, N_sep, N_token)
        bert_out = self.bert(**encoded_dict)[0] # (BATCH_SIZE, sequence_len, BERT_DIM)
        bert_tokens = bert_out[batch_indices, indices_by_batch, :]
        # bert_tokens: (batch_size, N_sep, N_token, BERT_dim)
        sentence_reps, sentence_mask = self.word_attention(bert_tokens, mask) 
        # (Batch_size, N_sep, BERT_DIM), (Batch_size, N_sep)
        
        # Prepare splitting
        indices = torch.arange(domain_indices.size(0))
        select_fever = domain_indices==0
        select_scifact = domain_indices==1
        
        fever_indices = indices[select_fever]
        scifact_indices = indices[select_scifact]
        original_indices = torch.cat([fever_indices, scifact_indices])
        
        # Split sentence_reps and sentence_mask
        fever_sentence_reps = sentence_reps[select_fever]
        fever_sentence_mask = sentence_mask[select_fever]

        scifact_sentence_reps = sentence_reps[select_scifact]
        scifact_sentence_mask = sentence_mask[select_scifact]
        
        if rationale_label is not None:
            fever_rationale_label = rationale_label[select_fever]
            scifact_rationale_label = rationale_label[select_scifact]
        
        # Compute rationale_out
        fever_rationale_out = self.rationale_linear_fever(fever_sentence_reps) # (Batch_size, N_sep, 2)
        scifact_rationale_out = self.rationale_linear_scifact(scifact_sentence_reps)
        
        fever_att_scores = fever_rationale_out[:,:,1] # (BATCH_SIZE, N_sentence)
        scifact_att_scores = scifact_rationale_out[:,:,1] # (BATCH_SIZE, N_sentence)
        
        if bool(torch.rand(1) < sample_p): # Choose sentence according to predicted rationale
            fever_valid_scores = fever_rationale_out[:,:,1] > fever_rationale_out[:,:,0]
            scifact_valid_scores = scifact_rationale_out[:,:,1] > scifact_rationale_out[:,:,0]
        else:
            fever_valid_scores = fever_rationale_label == 1 # Ground truth
            scifact_valid_scores = scifact_rationale_label == 1
        
        fever_paragraph_rep = self.sentence_attention_fever(fever_sentence_reps, 
                                                            fever_sentence_mask, fever_att_scores, fever_valid_scores) 
        # (BATCH_SIZE, BERT_DIM) 

        scifact_paragraph_rep = self.sentence_attention_scifact(scifact_sentence_reps, 
                                                                scifact_sentence_mask, scifact_att_scores, scifact_valid_scores) 
        # (BATCH_SIZE, BERT_DIM)
        
        fever_stance_out = self.stance_linear_fever(fever_paragraph_rep) # (Batch_size, 3)
        scifact_stance_out = self.stance_linear_scifact(scifact_paragraph_rep) # (Batch_size, 3)
        
        # Combine splitted ones to the original order
        stance_out = torch.cat([fever_stance_out, scifact_stance_out])
        stance_out = stance_out[original_indices]
        
        rationale_out = torch.cat([fever_rationale_out, scifact_rationale_out])
        rationale_out = rationale_out[original_indices]
        
        if stance_label is not None:
            stance_loss = self.stance_criterion(stance_out, stance_label)
        else:
            stance_loss = None
        if rationale_label is not None:
            rationale_loss = self.rationale_criterion(rationale_out.view(-1, self.rationale_label_size), 
                                                      rationale_label.view(-1)) # ignore index 2
        else:
            rationale_loss = None
            
        stance_out = torch.argmax(stance_out.cpu(), dim=-1).detach().numpy().tolist()

        rationale_pred = torch.argmax(rationale_out.cpu(), dim=-1) # (Batch_size, N_sep)
        rationale_out = [rationale_pred_paragraph[mask].detach().numpy().tolist() for rationale_pred_paragraph, mask in zip(rationale_pred, sentence_mask.bool())]
            
            
        return rationale_out, stance_out, rationale_loss, stance_loss

class StanceParagraphClassifier(nn.Module):
    def __init__(self, bert_path, bert_dim, dropout = 0.1, ignore_index=2):
        super(StanceParagraphClassifier, self).__init__()
        self.stance_label_size = 3
        self.ignore_index = 2
        self.bert = AutoModel.from_pretrained(bert_path)  
        self.stance_criterion = nn.CrossEntropyLoss()
        self.dropout = dropout
        self.bert_dim = bert_dim
        self.sentence_attention = DynamicSentenceAttention(bert_dim, bert_dim, dropout=dropout)
        self.word_attention = WordAttention(bert_dim, bert_dim, dropout=dropout)
        self.stance_linear = ClassificationHead(bert_dim, self.stance_label_size, hidden_dropout_prob = dropout)
        self.extra_modules = [
            self.sentence_attention,
            self.word_attention,
            self.stance_linear,
            self.stance_criterion,
        ]
            
    def reinitialize(self):
        self.extra_modules = []
        self.stance_linear = ClassificationHead(self.bert_dim, self.stance_label_size, hidden_dropout_prob = self.dropout)
        self.sentence_attention = DynamicSentenceAttention(self.bert_dim, self.bert_dim, dropout=self.dropout)
        self.extra_modules = [
            self.stance_linear,
            self.stance_criterion,
            self.word_attention,
            self.sentence_attention
        ]
    
    def forward(self, encoded_dict, transformation_indices, stance_label = None):
        batch_indices, indices_by_batch, mask = transformation_indices # (batch_size, N_sep, N_token)
        bert_out = self.bert(**encoded_dict)[0] # (BATCH_SIZE, sequence_len, BERT_DIM)
        bert_tokens = bert_out[batch_indices, indices_by_batch, :]
        # bert_tokens: (batch_size, N_sep, N_token, BERT_dim)
        sentence_reps, sentence_mask = self.word_attention(bert_tokens, mask) 
        # (Batch_size, N_sep, BERT_DIM), (Batch_size, N_sep)

        paragraph_rep = self.sentence_attention(sentence_reps, sentence_mask, sentence_mask.float(), sentence_mask) 
        # (BATCH_SIZE, BERT_DIM) 
        
        stance_out = self.stance_linear(paragraph_rep) # (Batch_size, 3)
        
        if stance_label is not None:
            stance_loss = self.stance_criterion(stance_out, stance_label)
        else:
            stance_loss = None
            
        stance_out = torch.argmax(stance_out.cpu(), dim=-1).detach().numpy().tolist()
          
        return stance_out, stance_loss
    
class RationaleParagraphClassifier(nn.Module):
    def __init__(self, bert_path, bert_dim, dropout = 0.1, ignore_index=2):
        super(RationaleParagraphClassifier, self).__init__()
        self.rationale_label_size = 2
        self.ignore_index = 2
        self.bert = AutoModel.from_pretrained(bert_path)  
        self.rationale_criterion = nn.CrossEntropyLoss(ignore_index = 2)
        self.dropout = dropout
        self.bert_dim = bert_dim
        self.sentence_attention = DynamicSentenceAttention(bert_dim, bert_dim, dropout=dropout)
        self.word_attention = WordAttention(bert_dim, bert_dim, dropout=dropout)
        self.rationale_linear = ClassificationHead(bert_dim, self.rationale_label_size, hidden_dropout_prob = dropout)
        self.extra_modules = [
            self.sentence_attention,
            self.word_attention,
            self.rationale_linear,
            self.rationale_criterion
        ]
            
    def reinitialize(self):
        self.extra_modules = []
        self.rationale_linear = ClassificationHead(self.bert_dim, self.rationale_label_size, hidden_dropout_prob = self.dropout)
        self.sentence_attention = DynamicSentenceAttention(self.bert_dim, self.bert_dim, dropout=self.dropout)
        self.extra_modules = [
            self.rationale_linear,
            self.rationale_criterion,
            self.word_attention,
            self.sentence_attention
        ]
    
    def forward(self, encoded_dict, transformation_indices, rationale_label = None, sample_p=1, rationale_score = False):
        batch_indices, indices_by_batch, mask = transformation_indices # (batch_size, N_sep, N_token)
        bert_out = self.bert(**encoded_dict)[0] # (BATCH_SIZE, sequence_len, BERT_DIM)
        bert_tokens = bert_out[batch_indices, indices_by_batch, :]
        # bert_tokens: (batch_size, N_sep, N_token, BERT_dim)
        sentence_reps, sentence_mask = self.word_attention(bert_tokens, mask) 
        # (Batch_size, N_sep, BERT_DIM), (Batch_size, N_sep)
        #print(bert_out.shape, bert_tokens.shape, sentence_reps.shape, sentence_mask.shape, rationale_label.shape)
        rationale_out = self.rationale_linear(sentence_reps) # (Batch_size, N_sep, 2)
        att_scores = rationale_out[:,:,1] # (BATCH_SIZE, N_sentence)
        
        if bool(torch.rand(1) < sample_p): # Choose sentence according to predicted rationale
            valid_scores = rationale_out[:,:,1] > rationale_out[:,:,0]
        else:
            valid_scores = rationale_label == 1 # Ground truth
        paragraph_rep = self.sentence_attention(sentence_reps, sentence_mask, att_scores, valid_scores) 
        # (BATCH_SIZE, BERT_DIM) 

        if rationale_label is not None:
            rationale_loss = self.rationale_criterion(rationale_out.view(-1, self.rationale_label_size), 
                                                      rationale_label.view(-1)) # ignore index 2
        else:
            rationale_loss = None
            
        if rationale_score:
            rationale_pred = rationale_out.cpu()[:,:,1] # (Batch_size, N_sep)
        else:
            rationale_pred = torch.argmax(rationale_out.cpu(), dim=-1) # (Batch_size, N_sep)
        rationale_out = [rationale_pred_paragraph[mask].detach().numpy().tolist() for rationale_pred_paragraph, mask in zip(rationale_pred, sentence_mask.bool())]            
        return rationale_out, rationale_loss