from .base import AbstractDataloader

import os
import torch
import random
import pickle
import numpy as np
import torch.utils.data as data_utils


def worker_init_fn(worker_id):
    random.seed(np.random.get_state()[1][0] + worker_id)                                                      
    np.random.seed(np.random.get_state()[1][0] + worker_id)


class LRUDataloader():
    def __init__(self, args, dataset):
        self.args = args
        self.rng = np.random
        self.save_folder = dataset._get_preprocessed_folder_path()
        dataset = dataset.load_dataset()
        self.train = dataset['train']
        self.val = dataset['val']
        self.test = dataset['test']
        self.umap = dataset['umap']
        self.smap = dataset['smap']
        self.user_count = len(self.umap)
        self.item_count = len(self.smap)

        args.num_users = self.user_count
        args.num_items = self.item_count
        self.max_len = args.bert_max_len
        self.sliding_size = args.sliding_window_size

    @classmethod
    def code(cls):
        return 'lru'

    def get_pytorch_dataloaders(self):
        train_loader = self._get_train_loader()
        val_loader = self._get_val_loader()
        test_loader = self._get_test_loader()
        return train_loader, val_loader, test_loader
    
    def get_pytorch_test_subset_dataloader(self):
        retrieved_file_path = self.args.llm_retrieved_path
        print('Loading retrieved file from {}'.format(retrieved_file_path))
        retrieved_file = pickle.load(open(os.path.join(retrieved_file_path,
                                                       'retrieved.pkl'), 'rb'))
        
        test_probs = retrieved_file['test_probs']
        test_labels = retrieved_file['test_labels']
        test_users = [u for u, (p, l) in enumerate(zip(test_probs, test_labels), start=1) \
                      if l in torch.topk(torch.tensor(p), self.args.llm_negative_sample_size+1).indices]

        dataset = dataset = LRUTestDataset(self.args, self.train, self.val, self.test, self.max_len, 
                                           self.rng, subset_users=test_users)
        dataloader = data_utils.DataLoader(dataset, batch_size=self.args.val_batch_size, shuffle=False,
                                           pin_memory=True, num_workers=self.args.num_workers)
        return dataloader

    def _get_train_loader(self):
        dataset = self._get_train_dataset()
        dataloader = data_utils.DataLoader(dataset, batch_size=self.args.train_batch_size,
                        shuffle=True, pin_memory=True, num_workers=self.args.num_workers,
                        worker_init_fn=worker_init_fn)
        return dataloader

    def _get_train_dataset(self):
        dataset = LRUTrainDataset(
            self.args, self.train, self.max_len, self.sliding_size, self.rng)
        return dataset

    def _get_val_loader(self):
        return self._get_eval_loader(mode='val')

    def _get_test_loader(self):
        return self._get_eval_loader(mode='test')

    def _get_eval_loader(self, mode):
        batch_size = self.args.val_batch_size if mode == 'val' else self.args.test_batch_size
        dataset = self._get_eval_dataset(mode)
        dataloader = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        pin_memory=True, num_workers=self.args.num_workers)
        return dataloader

    def _get_eval_dataset(self, mode):
        if mode == 'val':
            dataset = LRUValidDataset(self.args, self.train, self.val, self.max_len, self.rng)
        elif mode == 'test':
            dataset = LRUTestDataset(self.args, self.train, self.val, self.test, self.max_len, self.rng)
        return dataset


class LRUTrainDataset(data_utils.Dataset):
    def __init__(self, args, u2seq, max_len, sliding_size, rng):
        self.args = args
        self.max_len = max_len
        self.sliding_step = int(sliding_size * max_len)
        self.num_items = args.num_items
        self.rng = rng
        
        assert self.sliding_step > 0
        self.all_seqs = []
        for u in sorted(u2seq.keys()):
            seq = u2seq[u]
            if len(seq) < self.max_len + self.sliding_step:
                self.all_seqs.append(seq)
            else:
                start_idx = range(len(seq) - max_len, -1, -self.sliding_step)
                self.all_seqs = self.all_seqs + [seq[i:i + max_len] for i in start_idx]

    def __len__(self):
        return len(self.all_seqs)

    def __getitem__(self, index):
        seq = self.all_seqs[index]
        labels = seq[-self.max_len:]
        tokens = seq[:-1][-self.max_len:]

        mask_len = self.max_len - len(tokens)
        tokens = [0] * mask_len + tokens

        mask_len = self.max_len - len(labels)
        labels = [0] * mask_len + labels

        return torch.LongTensor(tokens), torch.LongTensor(labels)


class LRUValidDataset(data_utils.Dataset):
    def __init__(self, args, u2seq, u2answer, max_len, rng):
        self.args = args
        self.u2seq = u2seq
        self.u2answer = u2answer
        users = sorted(self.u2seq.keys())
        self.users = [u for u in users if len(u2answer[u]) > 0]
        self.max_len = max_len
        self.rng = rng
    
    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user]
        answer = self.u2answer[user]

        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq

        return torch.LongTensor(seq), torch.LongTensor(answer)


class LRUTestDataset(data_utils.Dataset):
    def __init__(self, args, u2seq, u2val, u2answer, max_len, rng, subset_users=None):
        self.args = args
        self.u2seq = u2seq
        self.u2val = u2val
        self.u2answer = u2answer
        users = sorted(self.u2seq.keys())
        self.users = [u for u in users if len(u2val[u]) > 0 and len(u2answer[u]) > 0]
        self.max_len = max_len
        self.rng = rng
        
        if subset_users is not None:
            self.users = subset_users

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user] + self.u2val[user]
        answer = self.u2answer[user]

        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq

        return torch.LongTensor(seq), torch.LongTensor(answer)