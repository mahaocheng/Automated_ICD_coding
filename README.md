# Automated_ICD_coding
The code for our paper 'Automated ICD Coding Based on Word Embedding with Entry Embedding and Attention Mechanism(基于融合条目词嵌入和注意力机制的自动ICD编码)'
## Data Description
  Because the data is private, it is not available for public. 
  Raw data is very large and complicated, so we should extract the text data that we need. The code for extracting text data is in the folder 'extract_data'.The text data includes 6 kinds of texts which have entry names (chief complaint, present illness history, examination report, first progress note, ward-round records, discharge records) and the label(ICD-10 diagnose code). All the chinese texts have been tokenized by jieba tokenizer. 
```
------------------------------ 
train data              60571  
validation data         7571
test data               7571
-------------------------------
```
## Model Description
The framework of our proposed model includes a two-layer BiLSTM text encoder and a full connected network classifier.
Text encoder includes three modules:
1. word embedding with entry embedding module;
2. keyword attention module;
3. word attention module.

See in the below picture:
![picture](https://github.com/zhanghk-pku/Automated_ICD_coding/blob/master/model.png)

### keyword attention module:
![picture](https://github.com/zhanghk-pku/Automated_ICD_coding/blob/master/keyword_attention.png)

## some relate works
1. Scheurwegs E, Luyckx K, Luyten L, et al. Assigning clinical codes with data-driven concept representation on Dutch clinical free text[J]. Journal of biomedical informatics, 2017, 69: 118-127.
2. Duarte F, Martins B, Pinto C S, et al. Deep neural models for ICD-10 coding of death certificates and autopsy reports in free-text[J]. Journal of biomedical informatics, 2018, 80: 64-77
3. Mullenbach J, Wiegreffe S, Duke J, et al. Explainable prediction of medical codes from clinical text[J]. arXiv preprint arXiv:1802.05695, 2018
4. Shi H, Xie P, Hu Z, et al. Towards automated icd coding using deep learning[J]. arXiv preprint arXiv:1711.04075, 2017.
5. Xie P, Xing E. A neural architecture for automated icd coding[C]//Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2018: 1066-1076.
6. Baumel T, Nassour-Kassis J, Cohen R, et al. Multi-label classification of patient notes: case study on ICD code assignment[C]//Workshops at the Thirty-Second AAAI Conference on Artificial Intelligence. 2018.
7. Xu K, Lam M, Pang J, et al. Multimodal Machine Learning for Automated ICD Coding[J]. arXiv preprint arXiv:1810.13348, 2018.


