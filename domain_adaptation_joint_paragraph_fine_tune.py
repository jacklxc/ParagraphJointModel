import argparse

import torch
import jsonlines
import os
import sys

import functools
print = functools.partial(print, flush=True)

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm
from typing import List
from sklearn.metrics import f1_score, precision_score, recall_score

import math
import random
import numpy as np

from tqdm import tqdm
from util import arg2param, flatten, stance2json, rationale2json, merge_json
from paragraph_model_dynamic import DomainAdaptationJointParagraphClassifier
from dataset import FEVERParagraphBatchDataset, SciFactParagraphBatchDataset, SciFact_FEVER_Dataset, Multiple_SciFact_Dataset

import logging

from lib.data import GoldDataset, PredictedDataset
from lib import metrics

def schedule_sample_p(epoch, total):
    return np.sin(0.5* np.pi* epoch / (total-1))

def reset_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def batch_rationale_label(labels, padding_idx = 2):
    max_sent_len = max([len(label) for label in labels])
    label_matrix = torch.ones(len(labels), max_sent_len) * padding_idx
    label_list = []
    for i, label in enumerate(labels):
        for j, evid in enumerate(label):
            label_matrix[i,j] = int(evid)
        label_list.append([int(evid) for evid in label])
    return label_matrix.long(), label_list

def predict(model, dataset):
    model.eval()
    rationale_predictions = []
    stance_preds = []
    
    def remove_dummy(rationale_out):
        return [out[1:] for out in rationale_out]

    with torch.no_grad():
        for batch in tqdm(DataLoader(dataset, batch_size = args.batch_size, shuffle=False)):
            encoded_dict = encode(tokenizer, batch)
            domain_indices = batch["dataset"].to(device)
            transformation_indices = token_idx_by_sentence(encoded_dict["input_ids"], 
                                                           tokenizer.sep_token_id, args.repfile)
            encoded_dict = {key: tensor.to(device) for key, tensor in encoded_dict.items()}
            transformation_indices = [tensor.to(device) for tensor in transformation_indices]
            rationale_out, stance_out, _, _ = model(encoded_dict, transformation_indices, domain_indices)
            stance_preds.extend(stance_out)                
            rationale_predictions.extend(remove_dummy(rationale_out))

    return rationale_predictions, stance_preds

def evaluation(model, dataset):
    model.eval()
    rationale_predictions = []
    rationale_labels = []
    stance_preds = []
    stance_labels = []
    
    def remove_dummy(rationale_out):
        return [out[1:] for out in rationale_out]

    with torch.no_grad():
        for batch in tqdm(DataLoader(dataset, batch_size = args.batch_size, shuffle=False)):
            encoded_dict = encode(tokenizer, batch)
            transformation_indices = token_idx_by_sentence(encoded_dict["input_ids"], 
                                                               tokenizer.sep_token_id, args.repfile)
            encoded_dict = {key: tensor.to(device) for key, tensor in encoded_dict.items()}
            transformation_indices = [tensor.to(device) for tensor in transformation_indices]
            stance_label = batch["stance"].to(device)
            domain_indices = batch["dataset"].to(device)
            padded_rationale_label, rationale_label = batch_rationale_label(batch["label"], padding_idx = 2)
            rationale_out, stance_out, rationale_loss, stance_loss = \
                model(encoded_dict, transformation_indices, domain_indices, stance_label = stance_label, 
                      rationale_label = padded_rationale_label.to(device))
            stance_preds.extend(stance_out)
            stance_labels.extend(stance_label.cpu().numpy().tolist())

            rationale_predictions.extend(remove_dummy(rationale_out))
            rationale_labels.extend(remove_dummy(rationale_label))

    stance_f1 = f1_score(stance_labels,stance_preds,average="micro",labels=[1, 2])
    stance_precision = precision_score(stance_labels,stance_preds,average="micro",labels=[1, 2])
    stance_recall = recall_score(stance_labels,stance_preds,average="micro",labels=[1, 2])
    rationale_f1 = f1_score(flatten(rationale_labels),flatten(rationale_predictions))
    rationale_precision = precision_score(flatten(rationale_labels),flatten(rationale_predictions))
    rationale_recall = recall_score(flatten(rationale_labels),flatten(rationale_predictions))
    return stance_f1, stance_precision, stance_recall, rationale_f1, rationale_precision, rationale_recall

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

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Train, cross-validate and run sentence sequence tagger")
    argparser.add_argument('--repfile', type=str, default = "roberta-large", help="Word embedding file")
    argparser.add_argument('--scifact_corpus', type=str, default="/nas/home/xiangcil/scifact/data/corpus.jsonl")
    argparser.add_argument('--scifact_train', type=str, default="/nas/home/xiangcil/CitationEvaluation/SciFact/claims_train_retrieved.jsonl")
    #argparser.add_argument('--scifact_train', type=str)
    argparser.add_argument('--pre_trained_model', type=str)
    argparser.add_argument('--scifact_test', type=str, default="/nas/home/xiangcil/CitationEvaluation/SciFact/claims_dev_retrieved.jsonl")
    argparser.add_argument('--dataset', type=str, default="/nas/home/xiangcil/CitationEvaluation/SciFact/claims_dev.jsonl")
    argparser.add_argument('--bert_lr', type=float, default=1e-5, help="Learning rate for BERT-like LM")
    argparser.add_argument('--lr', type=float, default=5e-6, help="Learning rate")
    argparser.add_argument('--dropout', type=float, default=0, help="embedding_dropout rate")
    argparser.add_argument('--bert_dim', type=int, default=1024, help="bert_dimension")
    argparser.add_argument('--epoch_start', type=int, default=0, help="Training epoch")
    argparser.add_argument('--epoch', type=int, default=10, help="Training epoch")
    argparser.add_argument('--MAX_SENT_LEN', type=int, default=512)
    argparser.add_argument('--loss_ratio', type=float, default=5)
    argparser.add_argument('--checkpoint', type=str, default = "domain_adaptation_roberta_joint_paragraph_fine_tune.model")
    argparser.add_argument('--log_file', type=str, default = "domain_adaptation_joint_paragraph_fine_tune_performances.jsonl")
    argparser.add_argument('--prediction', type=str, default = "prediction_domain_adaptation_roberta_joint_paragraph_dynamic_fine_tune.jsonl")
    argparser.add_argument('--update_step', type=int, default=10)
    argparser.add_argument('--batch_size', type=int, default=1) # roberta-large: 2; bert: 8
    argparser.add_argument('--scifact_k', type=int, default=0)

    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    reset_random_seed(12345)
    args = argparser.parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.repfile)

    if args.scifact_train:
        train = True
        #assert args.repfile is not None, "Word embedding file required for training."
    else:
        train = False
    if args.scifact_test:
        test = True
    else:
        test = False

    params = vars(args)

    for k,v in params.items():
        print(k,v)

    if train:
        train_set = SciFactParagraphBatchDataset(args.scifact_corpus, args.scifact_train, 
                                         sep_token = tokenizer.sep_token, k = args.scifact_k, downsample_n = 2)

    dev_set = SciFactParagraphBatchDataset(args.scifact_corpus, args.scifact_test, 
                                           sep_token = tokenizer.sep_token, k = args.scifact_k, downsample_n = 0)

    print("Loaded dataset!")

    model = DomainAdaptationJointParagraphClassifier(args.repfile, args.bert_dim, 
                                      args.dropout).to(device)

    if args.pre_trained_model is not None:
        model.load_state_dict(torch.load(args.pre_trained_model))
        print("Loaded pre-trained model.")

    if train:
        settings = [{'params': model.bert.parameters(), 'lr': args.bert_lr}]
        for module in model.extra_modules:
            settings.append({'params': module.parameters(), 'lr': args.lr})
        optimizer = torch.optim.Adam(settings)
        scheduler = get_cosine_schedule_with_warmup(optimizer, 0, args.epoch)

        #if torch.cuda.device_count() > 1:
        #    print("Let's use", torch.cuda.device_count(), "GPUs!")
        #    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        #    model = nn.DataParallel(model)

        model.train()
        prev_performance = 0
        for epoch in range(args.epoch_start, args.epoch):
            tq = tqdm(DataLoader(train_set, batch_size = args.batch_size, shuffle=True))
            for i, batch in enumerate(tq):
                encoded_dict = encode(tokenizer, batch)
                transformation_indices = token_idx_by_sentence(encoded_dict["input_ids"], tokenizer.sep_token_id, args.repfile)
                encoded_dict = {key: tensor.to(device) for key, tensor in encoded_dict.items()}
                transformation_indices = [tensor.to(device) for tensor in transformation_indices]
                stance_label = batch["stance"].to(device)
                domain_indices = batch["dataset"].to(device)
                padded_rationale_label, rationale_label = batch_rationale_label(batch["label"], padding_idx = 2)
                rationale_out, stance_out, rationale_loss, stance_loss = \
                    model(encoded_dict, transformation_indices, domain_indices, stance_label = stance_label, 
                          rationale_label = padded_rationale_label.to(device))
                rationale_loss *= args.loss_ratio
                loss = rationale_loss + stance_loss
                loss.sum().backward()

                if i % args.update_step == args.update_step - 1:
                    optimizer.step()
                    optimizer.zero_grad()
                    tq.set_description(f'Epoch {epoch}, iter {i}, loss: {round(loss.item(), 4)}, stance loss: {round(stance_loss.item(), 4)}, rationale loss: {round(rationale_loss.item(), 4)}')

            scheduler.step()

            # Evaluation
            train_score = evaluation(model, train_set)
            print(f'Epoch {epoch}, train stance f1 p r: %.4f, %.4f, %.4f, rationale f1 p r: %.4f, %.4f, %.4f' % train_score)

            dev_score = evaluation(model, dev_set)
            print(f'Epoch {epoch}, dev stance f1 p r: %.4f, %.4f, %.4f, rationale f1 p r: %.4f, %.4f, %.4f' % dev_score)

            dev_perf = dev_score[0] * dev_score[3]
            if dev_perf >= prev_performance:
                torch.save(model.state_dict(), args.checkpoint)
                best_state_dict = model.state_dict()
                prev_performance = dev_perf
                print("New model saved!")
            else:
                print("Skip saving model.")


    if test:
        if train:
            del model
            model = DomainAdaptationJointParagraphClassifier(args.repfile, args.bert_dim, 
                                                      args.dropout).to(device)
            model.load_state_dict(best_state_dict)
            print("Testing on the new model.")
        else:
            model.load_state_dict(torch.load(args.checkpoint))
            print("Loaded saved model.")

        # Evaluation
        #dev_score = evaluation(model, dev_set)
        #print(f'Test stance f1 p r: %.4f, %.4f, %.4f, rationale f1 p r: %.4f, %.4f, %.4f' % dev_score)

        rationale_predictions, stance_preds = predict(model, dev_set)
        rationale_json = rationale2json(dev_set.samples, rationale_predictions)
        stance_json = stance2json(dev_set.samples, stance_preds)
        stance_json = post_process_stance(rationale_json, stance_json)
        merged_json = merge_json(rationale_json, stance_json)

        with jsonlines.open(args.prediction, 'w') as output:
            for result in merged_json:
                output.write(result)

        data = GoldDataset(args.scifact_corpus, args.dataset)
        predictions = PredictedDataset(data, args.prediction)
        res = metrics.compute_metrics(predictions)
        params["evaluation"] = res
        with jsonlines.open(args.log_file, mode='a') as writer:
            writer.write(params)