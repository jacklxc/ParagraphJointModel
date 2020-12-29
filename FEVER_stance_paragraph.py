import sys
import argparse

import torch
import jsonlines
import os

import functools
print = functools.partial(print, flush=True)

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm
from typing import List
from sklearn.metrics import f1_score, precision_score, recall_score

import random
import numpy as np

from tqdm import tqdm
from util import arg2param, flatten, stance2json, rationale2json, merge_json
from paragraph_model_dynamic import StanceParagraphClassifier as JointParagraphClassifier
from dataset import FEVERStanceDataset as FEVERParagraphBatchDataset

import logging

def reset_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def evaluation(model, dataset):
    model.eval()
    stance_preds = []
    stance_labels = []

    with torch.no_grad():
        for batch in tqdm(DataLoader(dataset, batch_size = args.batch_size, shuffle=False)):
            encoded_dict = encode(tokenizer, batch)
            transformation_indices = token_idx_by_sentence(encoded_dict["input_ids"], 
                                                               tokenizer.sep_token_id, args.repfile)
            encoded_dict = {key: tensor.to(device) for key, tensor in encoded_dict.items()}
            transformation_indices = [tensor.to(device) for tensor in transformation_indices]
            stance_label = batch["stance"].to(device)
            stance_out, stance_loss = \
                model(encoded_dict, transformation_indices, stance_label = stance_label)
            stance_preds.extend(stance_out)
            stance_labels.extend(stance_label.cpu().numpy().tolist())

    stance_f1 = f1_score(stance_labels,stance_preds,average="micro",labels=[1, 2])
    stance_precision = precision_score(stance_labels,stance_preds,average="micro",labels=[1, 2])
    stance_recall = recall_score(stance_labels,stance_preds,average="micro",labels=[1, 2])
    return stance_f1, stance_precision, stance_recall


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

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Train, cross-validate and run sentence sequence tagger")
    argparser.add_argument('--repfile', type=str, default = "roberta-large", help="Word embedding file")
    argparser.add_argument('--train_file', type=str, default="/nas/home/xiangcil/scifact/data/fever_train_retrieved.jsonl")
    argparser.add_argument('--pre_trained_model', type=str)
    #argparser.add_argument('--train_file', type=str)
    argparser.add_argument('--test_file', type=str, default="/nas/home/xiangcil/scifact/data/fever_dev_retrieved.jsonl")
    argparser.add_argument('--bert_lr', type=float, default=5e-6, help="Learning rate for BERT-like LM")
    argparser.add_argument('--lr', type=float, default=1e-6, help="Learning rate")
    argparser.add_argument('--dropout', type=float, default=0, help="embedding_dropout rate")
    argparser.add_argument('--bert_dim', type=int, default=1024, help="bert_dimension")
    argparser.add_argument('--epoch', type=int, default=10, help="Training epoch")
    argparser.add_argument('--MAX_SENT_LEN', type=int, default=512)
    argparser.add_argument('--checkpoint', type=str, default = "fever_roberta_stance_paragraph")
    argparser.add_argument('--log_file', type=str, default = "fever_stance_paragraph_performances.jsonl")
    argparser.add_argument('--update_step', type=int, default=10)
    argparser.add_argument('--batch_size', type=int, default=1) # roberta-large: 2; bert: 8
    argparser.add_argument('--k', type=int, default=0)
    argparser.add_argument('--evaluation_step', type=int, default=50000)
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    
    reset_random_seed(12345)

    args = argparser.parse_args()
    
    with open(args.checkpoint+".log", 'w') as f:
        sys.stdout = f

        device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
        tokenizer = AutoTokenizer.from_pretrained(args.repfile)

        if args.train_file:
            train = True
            #assert args.repfile is not None, "Word embedding file required for training."
        else:
            train = False
        if args.test_file:
            test = True
        else:
            test = False

        params = vars(args)

        for k,v in params.items():
            print(k,v)

        if train:
            train_set = FEVERParagraphBatchDataset(args.train_file, 
                                                     sep_token = tokenizer.sep_token, k=args.k)
        dev_set = FEVERParagraphBatchDataset(args.test_file, 
                                               sep_token = tokenizer.sep_token, k=args.k)

        print("Loaded dataset!")

        model = JointParagraphClassifier(args.repfile, args.bert_dim, 
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

            for epoch in range(args.epoch):
                error_count = 0
                tq = tqdm(DataLoader(train_set, batch_size = args.batch_size, shuffle=True))
                for i, batch in enumerate(tq):
                    encoded_dict = encode(tokenizer, batch)
                    transformation_indices = token_idx_by_sentence(encoded_dict["input_ids"], tokenizer.sep_token_id, args.repfile)
                    encoded_dict = {key: tensor.to(device) for key, tensor in encoded_dict.items()}
                    transformation_indices = [tensor.to(device) for tensor in transformation_indices]
                    stance_label = batch["stance"].to(device)

                    stance_out, loss = \
                        model(encoded_dict, transformation_indices, stance_label = stance_label)
                    loss.sum().backward()

                    if i % args.update_step == args.update_step - 1:
                        optimizer.step()
                        optimizer.zero_grad()
                        tq.set_description(f'Epoch {epoch}, iter {i}, loss: {round(loss.item(), 4)}')
                        

                    if i % args.evaluation_step == args.evaluation_step-1:
                        torch.save(model.state_dict(), args.checkpoint+"_"+str(epoch)+"_"+str(i)+".model")
                        
                        # Evaluation
                        subset_train = Subset(train_set, range(0, 1000))
                        train_score = evaluation(model, subset_train)
                        print(f'Epoch {epoch}, step {i}, train stance f1 p r: %.4f, %.4f, %.4f' % train_score)

                        subset_dev = Subset(dev_set, range(0, 1000))
                        dev_score = evaluation(model, subset_dev)
                        print(f'Epoch {epoch}, step {i}, dev stance f1 p r: %.4f, %.4f, %.4f' % dev_score)
                scheduler.step()
                torch.save(model.state_dict(), args.checkpoint+"_"+str(epoch)+".model")
                print(error_count, "mismatch occurred.")

                # Evaluation
                subset_train = Subset(train_set, range(0, 10000))
                train_score = evaluation(model, subset_train)
                print(f'Epoch {epoch}, train stance f1 p r: %.4f, %.4f, %.4f' % train_score)

                subset_dev = Subset(dev_set, range(0, 10000))
                dev_score = evaluation(model, subset_dev)
                print(f'Epoch {epoch}, dev stance f1 p r: %.4f, %.4f, %.4f' % dev_score)



        if test:
            model = JointParagraphClassifier(args.repfile, args.bert_dim, 
                                              args.dropout).to(device)
            model.load_state_dict(torch.load(args.checkpoint))


            # Evaluation
            subset_dev = Subset(dev_set, range(0, 10000))
            dev_score = evaluation(model, subset_dev)
            print(f'Test stance f1 p r: %.4f, %.4f, %.4f' % dev_score)

            if train:
                params["stance_f1"] = dev_score[0]
                params["stance_precision"] = dev_score[1]
                params["stance_recall"] = dev_score[2]

                with jsonlines.open(args.log_file, mode='a') as writer:
                    writer.write(params)