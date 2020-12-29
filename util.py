import codecs
import numpy
import glob
import re
import numpy as np
from sklearn.metrics import f1_score

def flatten(arrayOfArray):
    array = []
    for arr in arrayOfArray:
        try:
            array.extend(arr)
        except:
            array.append(arr)
    return array

def read_passages(filename, is_labeled):
    str_seqs = []
    str_seq = []
    label_seqs = []
    label_seq = []
    for line in codecs.open(filename, "r", "utf-8"):
        lnstrp = line.strip()
        if lnstrp == "":
            if len(str_seq) != 0:
                str_seqs.append(str_seq)
                str_seq = []
                label_seqs.append(label_seq)
                label_seq = []
        else:
            if is_labeled:
                clause, label = lnstrp.split("\t")
                label_seq.append(label.strip())
            else:
                clause = lnstrp
            str_seq.append(clause)
    if len(str_seq) != 0:
        str_seqs.append(str_seq)
        str_seq = []
        label_seqs.append(label_seq)
        label_seq = []
    return str_seqs, label_seqs

def from_BIO_ind(BIO_pred, BIO_target, indices):
    table = {} # Make a mapping between the indices of BIO_labels and temporary original label indices
    original_labels = []
    for BIO_label,BIO_index in indices.items():
        if BIO_label[:2] == "I_" or BIO_label[:2] == "B_":
            label = BIO_label[2:]
        else:
            label = BIO_label
        if label in original_labels:
            table[BIO_index] = original_labels.index(label)
        else:
            table[BIO_index] = len(original_labels)
            original_labels.append(label)

    original_pred = [table[label] for label in BIO_pred]
    original_target = [table[label] for label in BIO_target]
    return original_pred, original_target

def to_BIO(label_seqs):
    new_label_seqs = []
    for label_para in label_seqs:
        new_label_para = []
        prev = ""
        for label in label_para:
            if label!="none": # "none" is O, remain unchanged.
                if label==prev:
                    new_label = "I_"+label
                else:
                    new_label = "B_"+label
            else:
                new_label = label # "none"
            prev = label
            new_label_para.append(new_label)
        new_label_seqs.append(new_label_para)
    return new_label_seqs

def from_BIO(label_seqs):
    new_label_seqs = []
    for label_para in label_seqs:
        new_label_para = []
        for label in label_para:
            if label[:2] == "I_" or label[:2] == "B_":
                new_label = label[2:]
            else:
                new_label = label
            new_label_para.append(new_label)
        new_label_seqs.append(new_label_para)
    return new_label_seqs

def clean_url(word):
    """
        Clean specific data format from social media
    """
    # clean urls
    word = re.sub(r'https? : \/\/.*[\r\n]*', '<URL>', word)
    word = re.sub(r'exlink', '<URL>', word)
    return word

def clean_num(word):
    # check if the word contain number and no letters
    if any(char.isdigit() for char in word):
        try:
            num = float(word.replace(',', ''))
            return '@'
        except:
            if not any(char.isalpha() for char in word):
                return '@'
    return word


def clean_words(str_seqs):
    processed_seqs = []
    for str_seq in str_seqs:
        processed_clauses = []
        for clause in str_seq:
            filtered = []
            tokens = clause.split()                 
            for word in tokens:
                word = clean_url(word)
                word = clean_num(word)
                filtered.append(word)
            filtered_clause = " ".join(filtered)
            processed_clauses.append(filtered_clause)
        processed_seqs.append(processed_clauses)
    return processed_seqs

def test_f1(test_file,pred_label_seqs):
    def linearize(labels):
        linearized = []
        for paper in labels:
            for label in paper:
                linearized.append(label)
        return linearized
    _, label_seqs = read_passages_original(test_file,True)
    true_label = linearize(label_seqs)
    pred_label = linearize(pred_label_seqs)

    f1 = f1_score(true_label,pred_label,average="weighted")
    print("F1 score:",f1)
    return f1
    
def postprocess(dataset, raw_flattened_output, raw_flattened_labels, MAX_SEQ_LEN):
    ground_truth_labels = []
    paragraph_lens = []
    for para in dataset.true_pairs:
        paragraph_lens.append(len(para["paragraph"]))
        ground_truth_labels.append(para["label"])

    raw_flattened_output = raw_flattened_output.tolist()
    raw_flattened_labels = raw_flattened_labels.tolist()
    batch_i = 0
    predicted_tags = []
    gt_tags = []
    for length in paragraph_lens:
        remaining_len = length
        predict_idx = []
        gt_tag = []
        while remaining_len > MAX_SEQ_LEN:
            this_batch = raw_flattened_output[batch_i*MAX_SEQ_LEN:(batch_i+1)*MAX_SEQ_LEN]
            this_batch_label = raw_flattened_labels[batch_i*MAX_SEQ_LEN:(batch_i+1)*MAX_SEQ_LEN]
            predict_idx.extend(this_batch)
            gt_tag.extend(this_batch_label)
            batch_i += 1
            remaining_len -= MAX_SEQ_LEN

        this_batch = raw_flattened_output[batch_i*MAX_SEQ_LEN:(batch_i+1)*MAX_SEQ_LEN]
        this_batch_label = raw_flattened_labels[batch_i*MAX_SEQ_LEN:(batch_i+1)*MAX_SEQ_LEN]
        predict_idx.extend(this_batch[:remaining_len])
        gt_tag.extend(this_batch_label[:remaining_len])
        predict_tag = [dataset.rev_label_ind[idx] for idx in predict_idx]
        gt_tag = [dataset.rev_label_ind[idx] for idx in gt_tag]
        batch_i += 1
        predicted_tags.append(predict_tag)
        gt_tags.append(gt_tag)
        
        
    predicted_tags = from_BIO(predicted_tags)
    final_gt = from_BIO(gt_tags)

    return predicted_tags, final_gt

def stance_postprocess(dataset, raw_output, raw_labels, MAX_SEQ_LEN):
    
    def combine(candidates):
        assert(len(candidates)>0)
        types = set(candidates)
        if len(types) == 1:
            return list(types)[0]
        elif 2 in types:
            return 2
        else:
            return 1
            
    
    ground_truth_labels = []
    paragraph_lens = []
    for para in dataset.true_pairs:
        paragraph_lens.append(len(para["paragraph"]))
        ground_truth_labels.append(para["label"])

    raw_output = raw_output.tolist()
    raw_labels = raw_labels.tolist()
    batch_i = 0
    predicted_tags = []
    gt_tags = []
    for length in paragraph_lens:
        remaining_len = length
        predict_idx = []
        gt_tag = []
        while remaining_len > MAX_SEQ_LEN:
            this_batch = raw_output[batch_i]
            this_batch_label = raw_labels[batch_i]
            predict_idx.append(this_batch)
            gt_tag.append(this_batch_label)
            batch_i += 1
            remaining_len -= MAX_SEQ_LEN

        this_batch = raw_output[batch_i]
        this_batch_label = raw_labels[batch_i]
        predict_idx.append(this_batch)
        gt_tag.append(this_batch_label)
        predict_tag = combine(predict_idx)
        gt_tag = combine(gt_tag)
        batch_i += 1
        predicted_tags.append(predict_tag)
        gt_tags.append(gt_tag)

    return predicted_tags, gt_tags

def rationale2json(true_pairs, predictions, excluded_pairs = None):
    claim_ids = []
    claims = {}
    assert(len(true_pairs) == len(predictions))
    for pair, prediction in zip(true_pairs, predictions):
        claim_id = pair["claim_id"]
        claim_ids.append(claim_id)
        
        predicted_sentences = []
        for i, pred in enumerate(prediction):
            if pred == "rationale" or pred == 1:
                predicted_sentences.append(i)

        this_claim = claims.get(claim_id, {"claim_id": claim_id, "evidence":{}})
        #if len(predicted_sentences) > 0:
        this_claim["evidence"][pair["doc_id"]] = predicted_sentences
        claims[claim_id] = this_claim
    if excluded_pairs is not None:
        for pair in excluded_pairs:
            claims[pair["claim_id"]] = {"claim_id": pair["claim_id"], "evidence":{}}
            claim_ids.append(pair["claim_id"])
    return [claims[claim_id] for claim_id in sorted(list(set(claim_ids)))]

def stance2json(true_pairs, predictions, excluded_pairs = None):
    claim_ids = []
    claims = {}
    idx2stance = ["NOT_ENOUGH_INFO", "SUPPORT", "CONTRADICT"]
    assert(len(true_pairs) == len(predictions))
    for pair, prediction in zip(true_pairs, predictions):
        claim_id = pair["claim_id"]
        claim_ids.append(claim_id)

        this_claim = claims.get(claim_id, {"claim_id": claim_id, "labels":{}})
        this_claim["labels"][pair["doc_id"]] = {"label": idx2stance[prediction], 'confidence': 1}
        claims[claim_id] = this_claim
    if excluded_pairs is not None:
        for pair in excluded_pairs:
            claims[pair["claim_id"]] = {"claim_id": pair["claim_id"], "labels":{}}
            claim_ids.append(pair["claim_id"])
    return [claims[claim_id] for claim_id in sorted(list(set(claim_ids)))]

def merge_json(rationale_jsons, stance_jsons):
    stance_json_dict = {str(stance_json["claim_id"]): stance_json for stance_json in stance_jsons}
    jsons = []
    for rationale_json in rationale_jsons:
        id = str(rationale_json["claim_id"])
        result = {}
        if id in stance_json_dict:
            for k, v in rationale_json["evidence"].items():
                if len(v) > 0 and stance_json_dict[id]["labels"][int(k)]["label"] is not "NOT_ENOUGH_INFO":
                    result[k] = {
                        "sentences": v,
                        "label": stance_json_dict[id]["labels"][int(k)]["label"]
                    }
        jsons.append({"id":int(id), "evidence": result})
    return jsons

def arg2param(args):
    params = vars(args)
    params["MAX_SEQ_LEN"]=params["CHUNK_SIZE"]*params["CHUNK_PER_SEQ"]
    params["MINIBATCH_SIZE"] = params["CHUNK_PER_SEQ"]
    params["SENTENCE_BATCH_SIZE"]=params["CHUNK_SIZE"]
    params["CHUNK_PER_STEP"]=params["PARAGRAPH_PER_STEP"]*params["CHUNK_PER_SEQ"]

    return params
