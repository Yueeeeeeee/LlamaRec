from datasets import dataset_factory

from .lru import *
from .llm import *
from .utils import *


def dataloader_factory(args):
    dataset = dataset_factory(args)
    if args.model_code == 'lru':
        dataloader = LRUDataloader(args, dataset)
    elif args.model_code == 'llm':
        dataloader = LLMDataloader(args, dataset)
    
    train, val, test = dataloader.get_pytorch_dataloaders()
    if 'llm' in args.model_code:
        tokenizer = dataloader.tokenizer
        test_retrieval = dataloader.test_retrieval
        return train, val, test, tokenizer, test_retrieval
    else:
        return train, val, test


def test_subset_dataloader_loader(args):
    dataset = dataset_factory(args)
    if args.model_code == 'lru':
        dataloader = LRUDataloader(args, dataset)
    elif args.model_code == 'llm':
        dataloader = LLMDataloader(args, dataset)

    return dataloader.get_pytorch_test_subset_dataloader()
