from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from scipy.spatial import distance
import json
import numpy as np
import jsonlines

import argparse

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--claim_file', type=str)
    argparser.add_argument('--corpus_file', type=str)
    argparser.add_argument('--k_retrieval', type=int)
    argparser.add_argument('--claim_retrieved_file', type=str)
    
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
    claims_by_id = {claim['id']:claim for claim in claims}
    
    corpus_texts = []
    corpus_ids = []
    for k, v in corpus.items():
        original_sentences = [v['title']] + v['abstract']
        processed_paragraph = " ".join(original_sentences)
        corpus_texts.append(processed_paragraph)
        corpus_ids.append(k)
    vectorizer = TfidfVectorizer(stop_words='english',
                                 ngram_range=(1, 2))
    corpus_ids = np.array(corpus_ids)
    corpus_vectors = vectorizer.fit_transform(corpus_texts)
    
    claim_vectors = vectorizer.transform([claim['claim'] for claim in claims])
    similarity_matrix = np.dot(corpus_vectors, claim_vectors.T).todense()
    
    k = args.k_retrieval
    orders = np.argsort(similarity_matrix,axis=0)
    retrieved_corpus = {claim["id"]: corpus_ids[orders[:,i][::-1][:k]].squeeze() for i, claim in enumerate(claims)} 
    
    with jsonlines.open(args.claim_retrieved_file, 'w') as output:
        claim_ids = sorted(list(claims_by_id.keys()))
        for id in claim_ids:
            claims_by_id[id]["retrieved_doc_ids"] = retrieved_corpus[id].tolist()
            output.write(claims_by_id[id])