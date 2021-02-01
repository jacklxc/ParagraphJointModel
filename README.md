# ParagraphJointModel
Implementation of The AAAI-21 Workshop on Scientific Document Understanding paper [A Paragraph-level Multi-task Learning Model for Scientific Fact-Verification](https://arxiv.org/abs/2012.14500). This work is at the top of [SciFact leaderboard](https://leaderboard.allenai.org/scifact/submissions/public) as of Jan 28th, 2021.

## Reproducing SciFact Leaderboard Result
### Dependencies

We recommend you create an anaconda environment:
```bash
conda create --name scifact python=3.6 conda-build
```
Then, from the `scifact` project root, run
```
conda develop .
```
which will add the scifact code to your `PYTHONPATH`.

Then, install Python requirements:
```
pip install -r requirements.txt
```
If you encounter any installation problem regarding sent2vec, please check [their repo](https://github.com/epfml/sent2vec).  
The BioSentVec model is available [here](https://github.com/ncbi-nlp/BioSentVec#biosentvec).

The SciFact claim files and corpus file are available at [SciFact repo](https://github.com/allenai/scifact).
The checkpoint of the model used for leaderboard submission is available [here](https://drive.google.com/file/d/1hMrQzFe1EaJpCN9s3pF27Wu3amBbekiI/view?usp=sharing).

### Abstract Retrieval
```
python ComputeBioSentVecAbstractEmbedding.py --claim_file /path/to/claims.jsonl --corpus_file /path/to/corpus.jsonl --sentvec_path /path/to/sentvec_model

python SentVecAbstractRetriaval.py --claim_file /path/to/claims.jsonl --corpus_file /path/to/corpus.jsonl --k_retrieval 30 --claim_retrieved_file /output/path/of/retrieval_file.jsonl --scifact_abstract_retrieval_file /output/path/of/retrieval_file_scifact_format.jsonl
```
The retrieved abstracts are available here: [train](https://drive.google.com/file/d/18yWhLP3n1OjT_XrUB3rJwNMnLRI3k8Ck/view?usp=sharing), [dev](https://drive.google.com/file/d/1fnfdOA2e3_U-kGavuhoyiYZUlDYWX9eM/view?usp=sharing), [test](https://drive.google.com/file/d/10Lh0aP06tGfZ-LlNGWnDtN0GM8M14z2q/view?usp=sharing).
### Training of the ParagraphJoint Model (Optional for Result Reproduction Purpose)
#### FEVER Pre-training
You need to retrieve some negative samples for FEVER pre-training. We used the trieval code from [here](https://github.com/sheffieldnlp/fever-naacl-2018). Empirically, only retrieving 5 negative examples for each claim is enough, while retrieving more may be way too time-consuming. You need to convert the format of the output of the retrieval code to the input of SciFact.

For your convenience, the converted retrieved FEVER examples with `k_retrieval=15` are available: [train](https://drive.google.com/file/d/1sS6mpaALuWnk6Pl2twIt_GcBs7ExRY2b/view?usp=sharing), [dev](https://drive.google.com/file/d/1sOfFL6fvK-AYjzcGPJ5KqcFPmAMvQJUi/view?usp=sharing).

Run `FEVER_joint_paragraph_dynamic.py` to pre-train the model on FEVER. Use `--checkpoint` to specify the checkpoint path. Run `scifact_joint_paragraph_dynamic.py` to fine-tune on SciFact dataset. Use `--pre_trained_model` to load the pre-trained model. Please check the other options in the source file.

### Joint Prediction of Rationale Selection and Stance Prediciton
```
python scifact_joint_paragraph_dynamic_prediction.py --corpus_file /path/to/corpus.jsonl --test_file /path/to/retrieval_file.jsonl --dataset /path/to/scifact/claims_test.jsonl --batch_size 25 --k 30 --prediction /path/to/output.jsonl --evaluate --checkpoint /path/to/checkpoint
```

## File naming conventions
The file names should be self-explanatory. Most parameters are set with default values. The parameters should be straight forward.

### Non-Joint Models
File names with `rationale` and `stance` are those scripts for rationale selection and stance prediction models.

### FEVER Pretraining and Domain-Adaptation
File names with `FEVER` are scripts for training on FEVER dataset. Same for `domain_adaptation`.

### Prediction
File names with `prediction` are scripts for taking the pre-trained models and perform inference.

### KGAT
File names with `kgat` means those models with [KGAT](https://github.com/xiangwang1223/knowledge_graph_attention_network) as stance predictor.

### Fine-tuning
You can use `--pre_trained_model path/to/pre_trained.model` to load a model trained on FEVER dataset and fine-tune on SciFact.


