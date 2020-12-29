# ParagraphJointModel
Implementation of The AAAI-21 Workshop on Scientific Document Understanding paper "A Paragraph-level Multi-task Learning Model for Scientific Fact-Verification".

## Requirements
* tqdm
* python3
* jsonlines
* nltk
* numpy
* pytorch (developed with version 1.6.0)
* scikit-learn
* sent2vec
* transformers (Huggingface)

## Abstract Retrieval
Please refer to `.ipynb` files in `abstract_retrieval`.

## Rationale Selection and Stance Prediction Models
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

## FEVER Pre-training
You need to retrieve some negative samples for FEVER pre-training. I used the trieval code from [here](https://github.com/sheffieldnlp/fever-naacl-2018). You need to convert the format of the output of the retrieval code to the input of SciFact.