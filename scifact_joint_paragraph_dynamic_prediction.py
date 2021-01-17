import argparse

import torch
import jsonlines
import os

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm
from typing import List
from sklearn.metrics import f1_score, precision_score, recall_score

import math
import random
import numpy as np

from tqdm import tqdm
from util import arg2param, flatten, stance2json, rationale2json, merge_json
from paragraph_model_dynamic import JointParagraphClassifier
from dataset import SciFactParagraphBatchDataset

import logging

from lib.data import GoldDataset, PredictedDataset
from lib import metrics

def reset_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def predict(model, dataset):
    model.eval()
    rationale_predictions = []
    stance_preds = []
    
    def remove_dummy(rationale_out):
        return [out[1:] for out in rationale_out]

    with torch.no_grad():
        for batch in tqdm(DataLoader(dataset, batch_size = args.batch_size, shuffle=False)):
            encoded_dict = encode(tokenizer, batch)
            transformation_indices = token_idx_by_sentence(encoded_dict["input_ids"], 
                                                           tokenizer.sep_token_id, args.repfile)
            encoded_dict = {key: tensor.to(device) for key, tensor in encoded_dict.items()}
            transformation_indices = [tensor.to(device) for tensor in transformation_indices]
            rationale_out, stance_out, _, _ = model(encoded_dict, transformation_indices)
            stance_preds.extend(stance_out)                
            rationale_predictions.extend(remove_dummy(rationale_out))

    return rationale_predictions, stance_preds


def encode(tokenizer, batch, max_sent_len = 512):
    def truncate(input_ids, max_length, sep_token_id, pad_token_id):
        def longest_first_truncation(sentences, objective):
            sent_lens = [len(sent) for sent in sentences]
            while np.sum(sent_lens) > objective:
                max_position = np.argmax(sent_lens)
                sent_lens[max_position] -= 1
            return [sentence[:length] for sentence, length in zip(sentences, sent_lens)]

        all_paragraphs = []
        for paragraph in input_ids:
            valid_paragraph = paragraph[paragraph != pad_token_id]
            if valid_paragraph.size(0) <= max_length:
                all_paragraphs.append(paragraph[:max_length].unsqueeze(0))
            else:
                sep_token_idx = np.arange(valid_paragraph.size(0))[(valid_paragraph == sep_token_id).numpy()]
                idx_by_sentence = []
                prev_idx = 0
                for idx in sep_token_idx:
                    idx_by_sentence.append(paragraph[prev_idx:idx])
                    prev_idx = idx
                objective = max_length - 1 - len(idx_by_sentence[0]) # The last sep_token left out
                truncated_sentences = longest_first_truncation(idx_by_sentence[1:], objective)
                truncated_paragraph = torch.cat([idx_by_sentence[0]] + truncated_sentences + [torch.tensor([sep_token_id])],0)
                all_paragraphs.append(truncated_paragraph.unsqueeze(0))

        return torch.cat(all_paragraphs, 0)

    inputs = zip(batch["claim"], batch["paragraph"])
    encoded_dict = tokenizer.batch_encode_plus(
        inputs,
        pad_to_max_length=True,add_special_tokens=True,
        return_tensors='pt')
    if encoded_dict['input_ids'].size(1) > max_sent_len:
        if 'token_type_ids' in encoded_dict:
            encoded_dict = {
                "input_ids": truncate(encoded_dict['input_ids'], max_sent_len, 
                                      tokenizer.sep_token_id, tokenizer.pad_token_id),
                'token_type_ids': encoded_dict['token_type_ids'][:,:max_sent_len],
                'attention_mask': encoded_dict['attention_mask'][:,:max_sent_len]
            }
        else:
            encoded_dict = {
                "input_ids": truncate(encoded_dict['input_ids'], max_sent_len, 
                                      tokenizer.sep_token_id, tokenizer.pad_token_id),
                'attention_mask': encoded_dict['attention_mask'][:,:max_sent_len]
            }
    return encoded_dict

def token_idx_by_sentence(input_ids, sep_token_id, model_name):
    """
    Compute the token indices matrix of the BERT output.
    input_ids: (batch_size, paragraph_len)
    batch_indices, indices_by_batch, mask: (batch_size, N_sentence, N_token)
    bert_out: (batch_size, paragraph_len,BERT_dim)
    bert_out[batch_indices,indices_by_batch,:]: (batch_size, N_sentence, N_token, BERT_dim)
    """
    padding_idx = -1
    sep_tokens = (input_ids == sep_token_id).bool()
    paragraph_lens = torch.sum(sep_tokens,1).numpy().tolist()
    indices = torch.arange(sep_tokens.size(-1)).unsqueeze(0).expand(sep_tokens.size(0),-1)
    sep_indices = torch.split(indices[sep_tokens],paragraph_lens)
    paragraph_lens = []
    all_word_indices = []
    for paragraph in sep_indices:
        if "roberta" in model_name:
            paragraph = paragraph[1:]
        word_indices = [torch.arange(paragraph[i]+1, paragraph[i+1]+1) for i in range(paragraph.size(0)-1)]
        paragraph_lens.append(len(word_indices))
        all_word_indices.extend(word_indices)
    indices_by_sentence = nn.utils.rnn.pad_sequence(all_word_indices, batch_first=True, padding_value=padding_idx)
    indices_by_sentence_split = torch.split(indices_by_sentence,paragraph_lens)
    indices_by_batch = nn.utils.rnn.pad_sequence(indices_by_sentence_split, batch_first=True, padding_value=padding_idx)
    batch_indices = torch.arange(sep_tokens.size(0)).unsqueeze(-1).unsqueeze(-1).expand(-1,indices_by_batch.size(1),indices_by_batch.size(-1))
    mask = (indices_by_batch>=0) 

    return batch_indices.long(), indices_by_batch.long(), mask.long()

def post_process_stance(rationale_json, stance_json):
    assert(len(rationale_json) == len(stance_json))
    for stance_pred, rationale_pred in zip(stance_json, rationale_json):
        assert(stance_pred["claim_id"] == rationale_pred["claim_id"])
        for doc_id, pred in rationale_pred["evidence"].items():
            if len(pred) == 0:
                stance_pred["labels"][doc_id]["label"] = "NOT_ENOUGH_INFO"
    return stance_json

def post_process_rationale_score(rationale_scores, max_positive=3): ### Doesn't seem to be helpful?
    def process_rationale_score(paragraph_rationale_scores):
        paragraph_rationale_scores = np.array(paragraph_rationale_scores)
        if np.sum(paragraph_rationale_scores > 0.5) > max_positive:
            output = np.zeros(paragraph_rationale_scores.shape)
            positive_indices = np.argsort(paragraph_rationale_scores)[::-1][:max_positive]
            output[positive_indices] = 1
        else:
            output = (paragraph_rationale_scores > 0.5).astype(int)
        return output.tolist()
    return [process_rationale_score(paragraph_rationale_scores) for paragraph_rationale_scores in rationale_scores]

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Train, cross-validate and run sentence sequence tagger")
    argparser.add_argument('--repfile', type=str, default = "roberta-large", help="Word embedding file")
    argparser.add_argument('--corpus_file', type=str, default="/home/xxl190027/scifact_data/corpus.jsonl")
    argparser.add_argument('--test_file', type=str, default="/home/xxl190027/scifact_data/claims_dev_retrieved.jsonl")
    argparser.add_argument('--dataset', type=str, default="/home/xxl190027/scifact_data/claims_dev.jsonl")
    argparser.add_argument('--dropout', type=float, default=0, help="embedding_dropout rate")
    argparser.add_argument('--bert_dim', type=int, default=1024, help="bert_dimension")
    argparser.add_argument('--MAX_SENT_LEN', type=int, default=512)
    argparser.add_argument('--checkpoint', type=str, default = "scifact_roberta_joint_paragraph.model")
    argparser.add_argument('--batch_size', type=int, default=25)
    argparser.add_argument('--k', type=int, default=0)
    argparser.add_argument('--rationale_selection', type=str)
    argparser.add_argument('--label_prediction', type=str)
    argparser.add_argument('--prediction', type=str)
    argparser.add_argument('--log_file', type=str, default = "prediction_dynamic.log")
    argparser.add_argument('--evaluate', action='store_true')
    
    
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    
    reset_random_seed(12345)
    
    args = argparser.parse_args()
    params = vars(args)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.repfile)

    dev_set = SciFactParagraphBatchDataset(args.corpus_file, args.test_file, 
                                           sep_token = tokenizer.sep_token, k = args.k, train=False)
    
    model = JointParagraphClassifier(args.repfile, args.bert_dim, 
                                      args.dropout).to(device)

    model.load_state_dict(torch.load(args.checkpoint))
    print("Loaded saved model.")
        
    reset_random_seed(12345)
    rationale_predictions, stance_preds = predict(model, dev_set)
    rationale_json = rationale2json(dev_set.samples, rationale_predictions)
    stance_json = stance2json(dev_set.samples, stance_preds)
    stance_json = post_process_stance(rationale_json, stance_json)
    merged_json = merge_json(rationale_json, stance_json)
    if args.rationale_selection is not None:
        with jsonlines.open(args.rationale_selection, 'w') as output:
            for result in rationale_json:
                output.write(result)
    if args.label_prediction is not None:          
        with jsonlines.open(args.label_prediction, 'w') as output:
            for result in stance_json:
                output.write(result)
    if args.prediction is not None:          
        with jsonlines.open(args.prediction, 'w') as output:
            for result in merged_json:
                output.write(result)
        if args.evaluate:
            data = GoldDataset(args.corpus_file, args.dataset)
            predictions = PredictedDataset(data, args.prediction)
            res = metrics.compute_metrics(predictions)
            params["evaluation"] = res
            with jsonlines.open(args.log_file, mode='a') as writer:
                writer.write(params)