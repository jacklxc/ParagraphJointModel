import jsonlines
import os
import torch
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
    
class SelectRationale(nn.Module):
    """
    input: (BATCH_SIZE, N_sentence, INPUT_SIZE)
    output: (BATCH_SIZE, INPUT_SIZE)
    """
    def __init__(self, hard_k):
        super(SelectRationale, self).__init__()
        self.hard_k = hard_k
        
    def forward(self, token_reps, token_mask, rationale_out):
        # token_reps: (BATCH_SIZE, N_sentence, N_token, INPUT_SIZE)
        # token_mask: (BATCH_SIZE, N_sentence, N_token)
        # rationale_out: (BATCH_SIZE, N_sentence, 2)
        att_scores = rationale_out[:,:,1] # (BATCH_SIZE, N_sentence)
        if token_reps.size(0) > 0:
            if self.hard_k > 0 and self.hard_k <= att_scores.size(1):
                top_att_scores, top_idx = torch.topk(att_scores, self.hard_k)
                batch_indices = torch.arange(top_idx.size(0)).unsqueeze(-1).expand(-1,top_idx.size(1))
                return token_reps[batch_indices, top_idx,:,:], token_mask[batch_indices, top_idx,:]
        return token_reps, token_mask
    
class DynamicRationale(nn.Module):
    """
    input: (BATCH_SIZE, N_sentence, INPUT_SIZE)
    output: (BATCH_SIZE, INPUT_SIZE)
    """
    def __init__(self):
        super(DynamicRationale, self).__init__()
        
    def forward(self, token_reps, token_mask, valid_sentences):
        # token_reps: (BATCH_SIZE, N_sentence, N_token, INPUT_SIZE)
        # token_mask: (BATCH_SIZE, N_sentence, N_token)
        # valid_sentences: (BATCH_SIZE, N_sentence)
    
        #valid_sentences = rationale_out[:,:,1] > rationale_out[:,:,0] # Only consider sentences predicted as rationales
        rationale_reps = token_reps[:,1:,:,:][valid_sentences]
        rationale_token_mask = token_mask[:,1:,:][valid_sentences]
        if len(rationale_reps.shape) == 3 or rationale_reps.size(1) == 0:
            rationale_reps = token_reps[:,1,:,:].unsqueeze(1) # First sentence is claim; second is dummy
            rationale_token_mask = token_mask[:,1,:].unsqueeze(1)
        return rationale_reps, rationale_token_mask
    
class KernelGraphAttentionNetwork(nn.Module):
    def __init__(self, bert_dim, kernel):
        super(KernelGraphAttentionNetwork, self).__init__()
        self.stance_label_size = 3
        
        self.mu = torch.tensor(self.kernal_mus(kernel), requires_grad = False).cuda() #to(device)
        self.sigma = torch.tensor(self.kernel_sigmas(kernel), requires_grad = False).cuda() 
        self.cos_similarity = nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.proj_select = nn.Linear(kernel, 1)
        self.proj_gat = nn.Sequential(
            nn.Linear(bert_dim * 2, 128),
            nn.ReLU(True),
            nn.Linear(128, 1)
        )
        self.proj_rationale = nn.Linear(kernel, 1)
        self.proj_label = nn.Linear(bert_dim*2, self.stance_label_size)
        
    def kernal_mus(self, n_kernels):
        """
        get the mu for each guassian kernel. Mu is the middle of each bin
        :param n_kernels: number of kernels (including exact match). first one is exact match
        :return: l_mu, a list of mu.
        """
        l_mu = [1]
        if n_kernels == 1:
            return l_mu

        bin_size = 2.0 / (n_kernels - 1)  # score range from [-1, 1]
        l_mu.append(1 - bin_size / 2)  # mu: middle of the bin
        for i in range(1, n_kernels - 1):
            l_mu.append(l_mu[i] - bin_size)
        return l_mu


    def kernel_sigmas(self, n_kernels):
        """
        get sigmas for each guassian kernel.
        :param n_kernels: number of kernels (including exactmath.)
        :param lamb:
        :param use_exact:
        :return: l_sigma, a list of simga
        """
        bin_size = 2.0 / (n_kernels - 1)
        l_sigma = [0.001]  # for exact match. small variance -> exact match
        if n_kernels == 1:
            return l_sigma

        l_sigma += [0.1] * (n_kernels - 1)
        return l_sigma
    
    def kernel_computation(self, input1, input2, mask, mu, sigma):
        """
        RBF kernel computation.
        intput1, input2: (batch_size, n_sentence, [n_sentence, ], n_token, n_token, bert_dim)
        mask: (batch_size, n_sentence, [n_sentence, ], n_token, n_token)
        mu: (1,1,[1,],1,1,kernel_size)
        sigma: (1,1,[1,],1,1,kernel_size)
        K: (batch_size, n_sentence, [n_sentence, ], n_token, kernel_size)
        """
        similarity = self.cos_similarity(input1.float(), input2.float())
        normalized_similarity = -0.5 * ((similarity.unsqueeze(-1) - mu) / sigma) ** 2
        pooling_sum = torch.sum(torch.exp(normalized_similarity) * mask.unsqueeze(-1).float() ,axis=-2)
        K = torch.log(torch.clamp(pooling_sum, min=1e-6))
        return K
    
    def edge_kernel(self, sentence_token_reps, token_mask):
        """
        Computs the stance prediction for each sentence.
        sentence_label_pred: (batch_size, n_sentence, label_size)
        """
        batch_size, n_sentence, n_token, n_rep = sentence_token_reps.shape
        z = sentence_token_reps[:,:,0,:]
        
        input1_exp = sentence_token_reps.unsqueeze(2).unsqueeze(4).expand(-1,-1,n_sentence, -1, n_token, -1)
        input2_exp = sentence_token_reps.unsqueeze(1).unsqueeze(3).expand(-1,n_sentence,-1, n_token, -1, -1)
        mu = self.mu.view(1,1,1,1,1,-1)
        sigma = self.sigma.view(1,1,1,1,1,-1)
        token_mask_exp = token_mask.unsqueeze(1).unsqueeze(-1).expand(-1, n_sentence, -1, -1, n_token) # (batch_size, n_sentence, n_sentence, n_token, n_token)
        K = self.kernel_computation(input1_exp, input2_exp, token_mask_exp, mu, sigma)
        
        token_mask_exp2 = token_mask.unsqueeze(1).expand(-1, n_sentence, -1, -1).unsqueeze(-1) # (batch_size, n_sentence, n_sentence, n_token, 1)
        proj_K = self.proj_select(K)        
        token_attention = torch.softmax(proj_K.masked_fill((~token_mask_exp2).bool(), -1e4), dim=-2)
        sentence_token_reps_exp = sentence_token_reps.unsqueeze(1).expand(-1,n_sentence,-1, -1, -1)
        z_hat = torch.sum(sentence_token_reps_exp * token_attention, axis=-2)
        z_exp = z.unsqueeze(1).expand(-1,n_sentence,-1, -1)
        z_cat = torch.cat([z_exp, z_hat],axis=-1) # (batch_size, n_sentence, n_sentence, 2 * bert_dim)
        beta = torch.softmax(self.proj_gat(z_cat),axis=1)
        v = torch.cat([torch.sum(beta * z_hat, axis=1), z], axis=-1)
        sentence_label_pred = torch.softmax(self.proj_label(v),axis=-1)
        
        return sentence_label_pred
    
    def node_kernel(self,claim_reps, sentence_token_reps, token_mask):
        """
        Computes the weight of each rationale to the final stance prediction.
        rationale_out: (batch_size, n_sentence, 1)
        """
        batch_size, n_sentence, n_token, n_rep = sentence_token_reps.shape
        claim_reps_exp = claim_reps.unsqueeze(1).unsqueeze(3).expand(-1,n_sentence,-1,n_token, -1)
        sentence_token_reps_exp = sentence_token_reps.unsqueeze(3).expand(-1,-1,-1, n_token,-1)
        mu = self.mu.view(1,1,1,1,-1)
        sigma = self.sigma.view(1,1,1,1,-1)
        token_mask_exp = token_mask.unsqueeze(-1).expand(-1,-1,-1,n_token) # (batch_size, n_sentence, n_token, n_token)
        K = self.kernel_computation(claim_reps_exp, sentence_token_reps_exp, token_mask_exp, mu, sigma)
        phi = torch.mean(K,axis=-2)
        rationale_out = torch.softmax(self.proj_rationale(phi),axis=1)
        return rationale_out
    
    def forward(self, claim_reps, sentence_token_reps, claim_token_mask, token_mask):
        """
        claim_reps: (batch_size, n_token, bert_dim)
        sentence_token_reps: (batch_size, n_sentence, n_token, bert_dim)
        claim_token_mask: (batch_size, n_token)
        token_mask: (batch_size, n_sentence, n_token)
        stance_pred: (batch_size, label_size)
        """
        #claim_reps = F.normalize(claim_reps, p=2, dim=-1)
        #sentence_token_reps = F.normalize(sentence_token_reps, p=2, dim=-1)
        
        sentence_label_pred = self.edge_kernel(sentence_token_reps, token_mask)
        rationale_out = self.node_kernel(claim_reps, sentence_token_reps, token_mask)
        
        stance_pred = torch.sum(sentence_label_pred * rationale_out, axis=1)
        return stance_pred
    
class JointParagraphKGATClassifier(nn.Module):
    def __init__(self, bert_path, bert_dim, dropout = 0.1, ignore_index=2, kernel=6):
        super(JointParagraphKGATClassifier, self).__init__()
        self.stance_label_size = 3
        self.rationale_label_size = 2
        self.ignore_index = 2
        self.kernel = kernel
        self.bert = AutoModel.from_pretrained(bert_path)  
        self.stance_criterion = nn.CrossEntropyLoss()
        self.rationale_criterion = nn.CrossEntropyLoss(ignore_index = 2)
        self.dropout = dropout
        self.bert_dim = bert_dim
        #self.reduced_bert_dim = 128
        #self.rationale_selector =  SelectRationale(3)
        self.rationale_selector = DynamicRationale()
        #self.kgat_linear = nn.Linear(self.bert_dim, self.reduced_bert_dim)
        self.rationale_linear = ClassificationHead(self.bert_dim, self.rationale_label_size, hidden_dropout_prob = dropout)
        self.kgat = KernelGraphAttentionNetwork(bert_dim, self.kernel)
        self.word_attention = WordAttention(bert_dim, bert_dim, dropout=dropout)
        self.extra_modules = [
            #self.kgat_linear,
            self.rationale_linear,
            self.stance_criterion,
            self.rationale_criterion,
            self.kgat,
            self.word_attention
        ]            
    
    def reinitialize(self):
        self.extra_modules = []
        #self.kgat_linear = nn.Linear(self.bert_dim, self.reduced_bert_dim)
        self.kgat = KernelGraphAttentionNetwork(self.bert_dim, self.kernel)
            
        self.extra_modules = [
            #self.kgat_linear,
            self.stance_criterion,
            self.rationale_criterion,
            self.word_attention,
            self.kgat
        ]
    
    def forward(self, encoded_dict, transformation_indices, stance_label = None, rationale_label = None, sample_p=1):
        batch_indices, indices_by_batch, mask = transformation_indices # (batch_size, N_sep, N_token)
        bert_out = self.bert(**encoded_dict)[0] # (BATCH_SIZE, sequence_len, BERT_DIM)
        bert_tokens = bert_out[batch_indices, indices_by_batch, :]
        # bert_tokens: (batch_size, N_sep, N_token, BERT_dim)
        sentence_reps, sentence_mask = self.word_attention(bert_tokens, mask) 
        # (Batch_size, N_sep, BERT_DIM), (Batch_size, N_sep)
        
        rationale_out = self.rationale_linear(sentence_reps[:,1:,:]) # (Batch_size, N_sep, 2) remove claim
        if bool(torch.rand(1) < sample_p): # Choose sentence according to predicted rationale
            valid_sentences = rationale_out[:,:,1] > rationale_out[:,:,0]
        else:
            valid_sentences = rationale_label == 1 # Ground truth
        #kgat_token_reps = self.kgat_linear(bert_tokens.view(-1, bert_tokens.size(-1))).view(bert_tokens.size(0), bert_tokens.size(1), bert_tokens.size(2), -1)
        kgat_token_reps = bert_tokens
        top_reps, top_mask = self.rationale_selector(kgat_token_reps, mask, valid_sentences)

        claim_reps = kgat_token_reps[:,0,:,:]
        stance_out = self.kgat(claim_reps, top_reps, mask[:,0,:], top_mask) # (Batch_size, 3)
                
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
        rationale_out = [rationale_pred_paragraph[mask[1:]].detach().numpy().tolist() for rationale_pred_paragraph, mask in zip(rationale_pred, sentence_mask.bool())] # Exclude claim mask
            
            
        return rationale_out, stance_out, rationale_loss, stance_loss
            
class DomainAdaptationJointParagraphKGATClassifier(nn.Module):
    def __init__(self, bert_path, bert_dim, dropout = 0.1, ignore_index=2, kernel=6):
        super(DomainAdaptationJointParagraphKGATClassifier, self).__init__()
        self.stance_label_size = 3
        self.rationale_label_size = 2
        self.ignore_index = 2
        self.kernel = kernel
        
        self.bert = AutoModel.from_pretrained(bert_path)  
        self.stance_criterion = nn.CrossEntropyLoss()
        self.rationale_criterion = nn.CrossEntropyLoss(ignore_index = 2)
        
        self.rationale_selector_fever = DynamicRationale()
        self.rationale_selector_scifact = DynamicRationale()
        
        self.kgat_fever = KernelGraphAttentionNetwork(bert_dim, self.kernel)
        self.kgat_scifact = KernelGraphAttentionNetwork(bert_dim, self.kernel)
        
        self.rationale_linear_fever = ClassificationHead(bert_dim, self.rationale_label_size, hidden_dropout_prob = dropout)
        self.rationale_linear_scifact = ClassificationHead(bert_dim, self.rationale_label_size, hidden_dropout_prob = dropout)

        self.word_attention = WordAttention(bert_dim, bert_dim, dropout=dropout)
        
        self.extra_modules = [
            self.word_attention,
            self.rationale_selector_fever,
            self.rationale_selector_scifact,
            self.kgat_fever,
            self.kgat_scifact,
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
        fever_bert_tokens = bert_tokens[select_fever]
        fever_mask = mask[select_fever]
        fever_sentence_reps = sentence_reps[select_fever]
        fever_sentence_mask = sentence_mask[select_fever]
        
        scifact_bert_tokens = bert_tokens[select_scifact]
        scifact_mask = mask[select_scifact]
        scifact_sentence_reps = sentence_reps[select_scifact]
        scifact_sentence_mask = sentence_mask[select_scifact]
        
        if rationale_label is not None:
            fever_rationale_label = rationale_label[select_fever]
            scifact_rationale_label = rationale_label[select_scifact]
        
        # Compute rationale_out
        fever_rationale_out = self.rationale_linear_fever(fever_sentence_reps[:,1:,:]) # (Batch_size, N_sep, 2)
        scifact_rationale_out = self.rationale_linear_scifact(scifact_sentence_reps[:,1:,:])
        
        fever_att_scores = fever_rationale_out[:,:,1] # (BATCH_SIZE, N_sentence)
        scifact_att_scores = scifact_rationale_out[:,:,1] # (BATCH_SIZE, N_sentence)
        
        if bool(torch.rand(1) < sample_p): # Choose sentence according to predicted rationale
            fever_valid_scores = fever_rationale_out[:,:,1] > fever_rationale_out[:,:,0]
            scifact_valid_scores = scifact_rationale_out[:,:,1] > scifact_rationale_out[:,:,0]
        else:
            fever_valid_scores = fever_rationale_label == 1 # Ground truth
            scifact_valid_scores = scifact_rationale_label == 1
        
        fever_top_reps, fever_top_mask = self.rationale_selector_fever(fever_bert_tokens, fever_mask, fever_valid_scores)
        scifact_top_reps, scifact_top_mask = self.rationale_selector_fever(scifact_bert_tokens, scifact_mask, scifact_valid_scores)
        
        fever_stance_out = self.kgat_fever(fever_bert_tokens[:,0,:,:], fever_top_reps, fever_mask[:,0,:], fever_top_mask) # (Batch_size, 3)
        scifact_stance_out = self.kgat_scifact(scifact_bert_tokens[:,0,:,:], scifact_top_reps, scifact_mask[:,0,:], scifact_top_mask) # (Batch_size, 3)
        
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
        rationale_out = [rationale_pred_paragraph[mask[1:]].detach().numpy().tolist() for rationale_pred_paragraph, mask in zip(rationale_pred, sentence_mask.bool())]
            
            
        return rationale_out, stance_out, rationale_loss, stance_loss
    
class KGATClassifier(nn.Module):
    def __init__(self, bert_path, bert_dim, dropout = 0.1, kernel=5):
        super(KGATClassifier, self).__init__()
        self.stance_label_size = 3
        self.kernel = kernel
        self.bert = AutoModel.from_pretrained(bert_path)  
        self.stance_criterion = nn.CrossEntropyLoss()
        self.dropout = dropout
        self.bert_dim = bert_dim
        #self.reduced_bert_dim = 128
        #self.kgat_linear = nn.Linear(self.bert_dim, self.reduced_bert_dim)
        self.kgat = KernelGraphAttentionNetwork(1024, self.kernel) ########
        self.extra_modules = [
            #self.kgat_linear,
            self.stance_criterion,
            self.kgat
        ]            
    
    def forward(self, encoded_dict, transformation_indices, stance_label = None):
        batch_indices, indices_by_batch, mask = transformation_indices # (batch_size, N_sep, N_token)
        bert_out = self.bert(**encoded_dict)[0] # (BATCH_SIZE, sequence_len, BERT_DIM)
        bert_tokens = bert_out[batch_indices, indices_by_batch, :]
        # bert_tokens: (batch_size, N_sep, N_token, BERT_dim)
        
        kgat_token_reps = bert_tokens
        #kgat_token_reps = self.kgat_linear(bert_tokens.view(-1, bert_tokens.size(-1))).view(bert_tokens.size(0), bert_tokens.size(1), bert_tokens.size(2), -1)

        claim_reps = kgat_token_reps[:,0,:,:]
        stance_out = self.kgat(claim_reps, kgat_token_reps[:,1:,:,:], mask[:,0,:], mask[:,1:,:]) # (Batch_size, 3)
                
        if stance_label is not None:
            stance_loss = self.stance_criterion(stance_out, stance_label)
        else:
            stance_loss = None
            
        stance_out = torch.argmax(stance_out.cpu(), dim=-1).detach().numpy().tolist()
            
        return stance_out, stance_loss