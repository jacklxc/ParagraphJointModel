from sklearn.metrics.pairwise import cosine_similarity
import pickle
import json
import numpy as np
from tqdm import tqdm
import jsonlines
import argparse
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--claim_file', type=str)
    argparser.add_argument('--corpus_file', type=str)
    argparser.add_argument('--k_retrieval', type=int)
    argparser.add_argument('--claim_retrieved_file', type=str)
    argparser.add_argument('--scifact_abstract_retrieval_file', type=str, help="abstract retreival in scifact format")
    argparser.add_argument('--corpus_embedding_pickle', type=str, default="corpus_paragraph_biosentvec.pkl")
    argparser.add_argument('--claim_embedding_pickle', type=str, default="claim_biosentvec.pkl")
    
    args = argparser.parse_args()
    
    with open(args.corpus_embedding_pickle,"rb") as f:
        corpus_embeddings = pickle.load(f)
        
    with open(args.claim_embedding_pickle,"rb") as f:
        claim_embeddings = pickle.load(f)
        
    claim_file = args.claim_file
    
    claims = []
    with open(claim_file) as f:
        for line in f:
            claim = json.loads(line)
            claims.append(claim)
    claims_by_id = {claim['id']:claim for claim in claims}
    
    all_similarities = {}
    for claim_id, claim_embedding in tqdm(claim_embeddings.items()):
        this_claim = {}
        for abstract_id, abstract_embedding in corpus_embeddings.items():
            claim_similarity = cosine_similarity(claim_embedding,abstract_embedding)
            this_claim[abstract_id] = claim_similarity
        all_similarities[claim_id] = this_claim
        
    ordered_corpus = {}
    for claim_id, claim_similarities in tqdm(all_similarities.items()):
        corpus_ids = []
        max_similarity = []
        for abstract_id, similarity in claim_similarities.items():
            corpus_ids.append(abstract_id)
            max_similarity.append(np.max(similarity))
        corpus_ids = np.array(corpus_ids)
        sorted_order = np.argsort(max_similarity)[::-1]
        ordered_corpus[claim_id] = corpus_ids[sorted_order]
        
    k = args.k_retrieval
    retrieved_corpus = {ID:v[:k] for ID,v in ordered_corpus.items()}
    
    with jsonlines.open(args.claim_retrieved_file, 'w') as output:
        claim_ids = sorted(list(claims_by_id.keys()))
        for id in claim_ids:
            claims_by_id[id]["retrieved_doc_ids"] = retrieved_corpus[id].tolist()
            output.write(claims_by_id[id])
            
    with jsonlines.open(args.scifact_abstract_retrieval_file, 'w') as output:
        claim_ids = sorted(list(claims_by_id.keys()))
        for id in claim_ids:
            doc_ids = retrieved_corpus[id].tolist()
            doc_ids = [int(id) for id in doc_ids]
            output.write({"claim_id": id, "doc_ids": doc_ids})
