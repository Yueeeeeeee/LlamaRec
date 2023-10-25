# LlamaRec: Two-Stage Recommendation using Large Language Models for Ranking

This repository is the PyTorch impelementation for the PGAI@CIKM 2023 paper **LlamaRec: Two-Stage Recommendation using Large Language Models for Ranking [[Paper](media/paper.pdf)]**

<img src=media/method.png width=1000>

We propose a two-stage framework using large language models for ranking-based recommendation (LlamaRec). In particular, we use small-scale sequential recommenders to retrieve candidates based on the user interaction history. Then, both history and retrieved items are fed to the LLM in text via a carefully designed prompt template. Instead of generating next-item titles, we adopt a verbalizer-based approach that transforms output logits into probability distributions over the candidate items. Therefore, LlamaRec can efficiently rank items without generating long text and achieve superior performance in both recommendation performance and efficiency.


## Requirements

Pytorch, transformers, peft, bitsandbytes etc. For our detailed running environment see requirements.txt


## How to run LRURec
The command below starts the training of our retriever model LRURec
```bash
python train_retriever.py
```
You will be prompted to select dataset from ML-100k, Beauty and Games. Once trainin is finished, evaluation is automatically performed with the best retriever model

Then, run the following command to train our ranker model based on Llama 2
```bash
python train_ranker.py --llm_retrieved_path PATH_TO_RETRIEVER
```
Please specify PATH_TO_RETRIEVER with the retriever path from the previous step. Similarly, evaluation is performed after training is finished. All weights and results are saved under ./experiments


## Performance

The table below reports our main performance results, with best results marked in bold and second best results underlined. For training and evaluation details, please refer to our paper

<img src=media/performance.png width=1000>


## Citation
Please consider citing the following paper if you use our methods in your research:
```bib
@article{yue2023llamarec,
  title={LlamaRec: Two-Stage Recommendation using Large Language Models for Ranking},
  author={Zhenrui Yue, Sara Rabhi, Gabriel de Souza Pereira Moreira, Dong Wang and Even Oldridge},
  journal={arXiv preprint arXiv},
  year={2023}
}
```