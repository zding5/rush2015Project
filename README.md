## Original work: A Neural Attention Model for Abstractive Sentence Summarization

Further development on the paper A Neural Attention Model for Abstractive Sentence Summarization (EMNLP 2015)[1].

Official code: https://github.com/facebookarchive/NAMAS

## Base repo
Code: https://github.com/toru34/rush_emnlp_2015

### 1. Requirements
- Python 3.6.0+
- DyNet 2.0+
- NumPy 1.12.1+
- scikit-learn 0.19.0+
- tqdm 4.15.0+
- Dynet Installation GPU version: http://dynet.readthedocs.io/en/latest/python.html; Must follow the manual installation instruction, otherwise it wouldn't be GPU compatible.

### 2. Prepare dataset
To get preprocedded gigaword corpus, run
```
sh download_data.sh
```
.

### 3. Train
#### Arguments
- `--gpu`: GPU ID to use. For cpu, set `-1` [default: `0`]
- `--n_epochs`: Number of epochs [default: `3`]
- `--n_train`: Number of training data (up to `3803957`) [default: `3803957`]
- `--n_valid`: Number of validation data (up to `189651`) [default: `189651`]
- `--batch_size`: Mini batch size [default: `32`]
- `--vocab_size`: Vocabulary size [default: `60000`]
- `--emb_dim`: Embedding size [default: `256`]
- `--hid_dim`: Hidden state size [default: `256`]
- `--encoder_type`: Encoder type. [default: `attention`]
    - `bow`: Bag-of-words encoder.
    - `attention`: Attention-based encoder.
- `--c`: Window size in neural language model [default: `5`]
- `--q`: Window size in attention-based encoder [default: `2`]
- `--alloc_mem`: Amount of memory to allocate [mb] [default: `8192`]

#### Command example
```
python train.py --n_epochs 10
```

### 4. Test
#### Arguments
- `--gpu`: GPU ID to use. For cpu, set `-1` [default: `0`]
- `--n_test`: Number of test data [default: `189651`]
- `--beam_size`: Beam size [default: `5`]
- `--max_len`: Maximum length of decoding [default: `100`]
- `--model_file`: Trained model file path [default: `./model_e1`]
- `--input_file`: Test file path [default: `./data/valid.article.filter.txt`]
- `--output_file`: Output file path [default: `./pred_y.txt`]
- `--w2i_file`: Word2Index file path [default: `./w2i.dump`]
- `--i2w_file`: Index2Word file path [default: `./i2w.dump`]
- `--alloc_mem`: Amount of memory to allocate [mb] [default: `1024`]

#### Command example
```
python test.py --beam_size 10
```

### 5. Evaluate
You can use pythonrouge[2] to compute the ROUGE scores.
An example is in `evaluate.ipynb`.

### References
- [1] A. M. Rush et al. 2015. A Neural Attention Model for Sentence Summarization. In Proceedings of EMNLP 2015 \[[pdf\]](https://aclweb.org/anthology/D/D15/D15-1044.pdf)
- [2] pythonrouge: https://github.com/tagucci/pythonrouge
