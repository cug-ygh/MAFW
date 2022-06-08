

[![](https://badgen.net/badge/license/MIT/green)](#License)
[![](https://badgen.net/badge/contact/yinhao2000/purple)](https://yinhao2000.github.io/)

# MAFW


MAFW: A Large-scale, Multi-modal, Compound Affective Database for Dynamic Facial Expression  Recognition in the Wild. The data source of MAFW is richer, not only from movies and TV dramas, but also includes short videos from reality shows, talk shows, news, variety shows, etc, which is closer to the real-world emotion.

### Features

- 10,045 video-audio clips.
- Each clip is annotated with a compound emotional category and a couple of sentences that describe the subjects’ affective behaviors in the clip.
- multiple modalities in the wild.
- 11 single-emotion categories and 32 multi-emotion categories.


## 1. Samples of MAFW


### 1.1 Examples of the single expressions in MAFW

![image](https://github.com/yinhao2000/MAFW/blob/main/imgs/image03.png)

### 1.2 Examples of the multiple expressions in MAFW
![image](https://github.com/yinhao2000/MAFW/blob/main/imgs/image04.png)


## 2. Datasets

> **Note:** We have set out some of the terms and conditions in this section, as detailed in the [academics document](https://github.com/yinhao2000/MAFW/blob/main/academics/mafw%20academics-final.pdf).

### 2.1 Some term&conditions
- MAFW database is available for non-commercial research purposes only.
- You agree not to reproduce, duplicate, copy, sell, trade, resell or exploit for commercial purposes, any portion of the clips, and any derived data.
- You agree not to further copy, publish, or distribute any portion of the MAFW database. Except for internal use at a single site within the same organization, it is allowed to make copies of the dataset.



### 2.2 How to get the MAFW Dataset



SHA-256 for feature files:

```text
`MOSI/Processed/unaligned_50.pkl`:  `78e0f8b5ef8ff71558e7307848fc1fa929ecb078203f565ab22b9daab2e02524`
`MOSI/Processed/aligned_50.pkl`:    `d3994fd25681f9c7ad6e9c6596a6fe9b4beb85ff7d478ba978b124139002e5f9`
`MOSEI/Processed/unaligned_50.pkl`: `ad8b23d50557045e7d47959ce6c5b955d8d983f2979c7d9b7b9226f6dd6fec1f`
`MOSEI/Processed/aligned_50.pkl`:   `45eccfb748a87c80ecab9bfac29582e7b1466bf6605ff29d3b338a75120bf791`
`SIMS/Processed/unaligned_39.pkl`:  `c9e20c13ec0454d98bb9c1e520e490c75146bfa2dfeeea78d84de047dbdd442f`
```

MMSA uses feature files that are organized as follows:

```python
{
    "train": {
        "raw_text": [],              # raw text
        "audio": [],                 # audio feature
        "vision": [],                # video feature
        "id": [],                    # [video_id$_$clip_id, ..., ...]
        "text": [],                  # bert feature
        "text_bert": [],             # word ids for bert
        "audio_lengths": [],         # audio feature lenth(over time) for every sample
        "vision_lengths": [],        # same as audio_lengths
        "annotations": [],           # strings
        "classification_labels": [], # Negative(0), Neutral(1), Positive(2). Deprecated in v_2.0
        "regression_labels": []      # Negative(<0), Neutral(0), Positive(>0)
    },
    "valid": {***},                  # same as "train"
    "test": {***},                   # same as "train"
}
```




## 3. Supported MSA Models

|    Type     |                   Model Name                            |                                          From                                          |    Published     |
| :---------: | :-----------------------------------------------------: | :------------------------------------------------------------------------------------: | :---------------: |
| Single-Task |        [TFN](src/MMSA/models/singleTask/TFN.py)         |        [Tensor-Fusion-Network](https://github.com/A2Zadeh/TensorFusionNetwork)         | EMNLP 2017    |
| Single-Task |    [EF_LSTM](src/MMSA/models/singleTask/EF_LSTM.py)     |               [MultimodalDNN](https://github.com/rhoposit/MultimodalDNN)               | ACL 2018 Workshop |
| Single-Task |     [LF_DNN](src/MMSA/models/singleTask/LF_DNN.py)      |               [MultimodalDNN](https://github.com/rhoposit/MultimodalDNN)               | ACL 2018 Workshop |
| Single-Task |        [LMF](src/MMSA/models/singleTask/LMF.py)         | [Low-rank-Multimodal-Fusion](https://github.com/Justin1904/Low-rank-Multimodal-Fusion) | ACL 2018          |
| Single-Task |        [MFN](src/MMSA/models/singleTask/MFN.py)         |               [Memory-Fusion-Network](https://github.com/pliang279/MFN)                | AAAI 2018          |
| Single-Task |  [Graph-MFN](src/MMSA/models/singleTask/Graph_MFN.py)   |    [Graph-Memory-Fusion-Network](https://github.com/A2Zadeh/CMU-MultimodalSDK/)        | ACL 2018          |
| Single-Task | [MulT](src/MMSA/models/singleTask/MulT.py)(without CTC) |      [Multimodal-Transformer](https://github.com/yaohungt/Multimodal-Transformer)      | ACL 2019          |
| Single-Task |        [MFM](src/MMSA/models/singleTask/MFM.py)         |                     [MFM](https://github.com/pliang279/factorized/)                    | ICRL 2019          |
| Multi-Task  |     [MLF_DNN](src/MMSA/models/multiTask/MLF_DNN.py)     |                         [MMSA](https://github.com/thuiar/MMSA)                         | ACL 2020          |
| Multi-Task  |        [MTFN](src/MMSA/models/multiTask/MTFN.py)        |                         [MMSA](https://github.com/thuiar/MMSA)                         | ACL 2020          |
| Multi-Task  |        [MLMF](src/MMSA/models/multiTask/MLMF.py)        |                         [MMSA](https://github.com/thuiar/MMSA)                         | ACL 2020          |
| Single-Task |   [BERT-MAG](src/MMSA/models/singleTask/BERT_MAG.py)    |        [MAG-BERT](https://github.com/WasifurRahman/BERT_multimodal_transformer)        | ACL 2020          |
| Single-Task |       [MISA](src/MMSA/models/singleTask/MISA.py)        |                      [MISA](https://github.com/declare-lab/MISA)                       | ACMMM 2020    |
| Single-Task |     [SELF_MM](src/MMSA/models/multiTask/SELF_MM.py)     |                      [Self-MM](https://github.com/thuiar/Self-MM)                      | AAAI 2021          |
| Single-Task |       [MMIM](src/MMSA/models/singleTask/MMIM.py)        |            [MMIM](https://github.com/declare-lab/Multimodal-Infomax)                   | EMNLP 2021    |
| Single-Task |           BBFN (Work in Progress)                       |               [BBFN](https://github.com/declare-lab/BBFN)                              | ICMI 2021          |


## 4. Results

Baseline results are reported in [results/result-stat.md](results/result-stat.md)

## 5. Citation

- [MAFW: A Large-scale, Multi-modal, Compound Affective Database for Dynamic Facial Expression Recognition in the Wild]()

Please cite our paper if you find our work useful for your research:

```
```
