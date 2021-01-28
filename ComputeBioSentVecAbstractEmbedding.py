import sent2vec
from nltk import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from scipy.spatial import distance
import glob
import pickle
import json

import argparse

def preprocess_sentence(text):
    text = text.replace('/', ' / ')
    text = text.replace('.-', ' .- ')
    text = text.replace('.', ' . ')
    text = text.replace('\'', ' \' ')
    text = text.lower()

    tokens = [token for token in word_tokenize(text) if token not in punctuation and token not in stop_words]

    return ' '.join(tokens)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--claim_file', type=str)
    argparser.add_argument('--corpus_file', type=str)
    argparser.add_argument('--sentvec_path', type=str)
    argparser.add_argument('--corpus_embedding_pickle', type=str, default="corpus_paragraph_biosentvec.pkl")
    argparser.add_argument('--claim_embedding_pickle', type=str, default="claim_biosentvec.pkl")

    args = argparser.parse_args()
    claim_file = args.claim_file
    corpus_file = args.corpus_file
    
    corpus = {}
    with open(corpus_file) as f:
        for line in f:
            abstract = json.loads(line)
            corpus[str(abstract["doc_id"])] = abstract
            
    claims = []
    with open(claim_file) as f:
        for line in f:
            claim = json.loads(line)
            claims.append(claim)
    
    model_path = args.sentvec_path
    model = sent2vec.Sent2vecModel()
    try:
        model.load_model(model_path)
    except Exception as e:
        print(e)
    print('model successfully loaded')
    
    stop_words = set(stopwords.words('english'))
    
    # By paragraph embedding
    corpus_embeddings = {}
    for k, v in corpus.items():
        original_sentences = [v['title']] + v['abstract']
        processed_paragraph = " ".join([preprocess_sentence(sentence) for sentence in original_sentences])
        sentence_vector = model.embed_sentence(processed_paragraph)
        corpus_embeddings[k] = sentence_vector
        
    with open(args.corpus_embedding_pickle,"wb") as f:
        pickle.dump(corpus_embeddings,f)
        
    claim_embeddings = {}
    for claim in claims:
        processed_sentence = preprocess_sentence(claim['claim'])
        sentence_vector = model.embed_sentence(processed_sentence)
        claim_embeddings[claim["id"]] = sentence_vector
        
    with open(args.claim_embedding_pickle,"wb") as f:
        pickle.dump(claim_embeddings,f)