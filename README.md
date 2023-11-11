# LlamaRec: Two-Stage Recommendation using Large Language Models for Ranking

This repository is the PyTorch impelementation for the PGAI@CIKM 2023 paper **LlamaRec: Two-Stage Recommendation using Large Language Models for Ranking [[Paper](https://arxiv.org/abs/2311.02089)]**.

<img src=media/method.png width=1000>

We propose a two-stage framework using large language models for ranking-based recommendation (LlamaRec). In particular, we use small-scale sequential recommenders to retrieve candidates based on the user interaction history. Then, both history and retrieved items are fed to the LLM in text via a carefully designed prompt template. Instead of generating next-item titles, we adopt a verbalizer-based approach that transforms output logits into probability distributions over the candidate items. Therefore, LlamaRec can efficiently rank items without generating long text and achieve superior performance in both recommendation performance and efficiency.


## Requirements

Pytorch, transformers, peft, bitsandbytes etc. For our detailed running environment see requirements.txt.


## How to run LlamaRec
The command below starts the training of the retriever model LRURec
```bash
python train_retriever.py
```
You can set additional arguments like weight_decay to change the hyperparameters. Upon the command, you will be prompted to select dataset from ML-100k, Beauty and Games. Once training is finished, evaluation is automatically performed with the best retriever model.

Then, run the following command to train the ranker model based on Llama 2
```bash
python train_ranker.py --llm_retrieved_path PATH_TO_RETRIEVER
```
Please specify PATH_TO_RETRIEVER with the retriever path from the previous step. To run this command, you will need access to meta-llama/Llama-2-7b-hf on the HF hub. Similarly, evaluation is performed after training is finished. All weights and results are saved under ./experiments.


## Performance

The table below reports our main performance results, with best results marked in bold and second best results underlined. For training and evaluation details, please refer to our paper.

<img src=media/performance.png width=1000>


## Citation
Please consider citing the following papers if you use our methods in your research:
```
@article{yue2023linear,
  title={Linear Recurrent Units for Sequential Recommendation},
  author={Yue, Zhenrui and Wang, Yueqi and He, Zhankui and Zeng, Huimin and McAuley, Julian and Wang, Dong},
  journal={arXiv preprint arXiv:2310.02367},
  year={2023}
}

@article{yue2023llamarec,
  title={LlamaRec: Two-Stage Recommendation using Large Language Models for Ranking},
  author={Yue, Zhenrui and Rabhi, Sara and Moreira, Gabriel de Souza Pereira and Wang, Dong and Oldridge, Even},
  journal={arXiv preprint arXiv:2311.02089},
  year={2023}
}
```
