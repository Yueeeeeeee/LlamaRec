import os
import torch
from abc import ABCMeta, abstractmethod


def save_state_dict(state_dict, path, filename):
    torch.save(state_dict, os.path.join(path, filename))


class LoggerService(object):
    def __init__(self, args, writer, val_loggers, test_loggers, use_wandb):
        self.args = args
        self.writer = writer
        self.val_loggers = val_loggers if val_loggers else []
        self.test_loggers = test_loggers if test_loggers else []
        self.use_wandb = use_wandb

    def complete(self):
        if self.use_wandb:
            self.writer.finish()
        else:
            self.writer.close()

    def log_val(self, log_data):
        criteria_met = False
        for logger in self.val_loggers:
            logger.log(self.writer, **log_data)
            if self.args.early_stopping and isinstance(logger, BestModelLogger):
                criteria_met = logger.patience_counter >= self.args.early_stopping_patience
        return criteria_met
    
    def log_test(self, log_data):
        for logger in self.test_loggers:
            logger.log(self.writer, **log_data)


class AbstractBaseLogger(metaclass=ABCMeta):
    @abstractmethod
    def log(self, *args, **kwargs):
        raise NotImplementedError

    def complete(self, *args, **kwargs):
        pass


class MetricGraphPrinter(AbstractBaseLogger):
    def __init__(self, key, graph_name, group_name, use_wandb):
        self.key = key
        self.graph_label = graph_name
        self.group_name = group_name
        self.use_wandb = use_wandb
        
    def log(self, writer, *args, **kwargs):
        if self.key in kwargs:
            if self.use_wandb:
                writer.log({self.group_name+'/'+self.graph_label: kwargs[self.key], 'batch': kwargs['accum_iter']})
            else:
                writer.add_scalar(self.group_name+'/'+ self.graph_label, kwargs[self.key], kwargs['accum_iter'])
        else:
            print('Metric {} not found...'.format(self.key))

    def complete(self, writer, *args, **kwargs):
        self.log(writer, *args, **kwargs)


class RecentModelLogger(AbstractBaseLogger):
    def __init__(self, args, checkpoint_path, filename='checkpoint-recent.pth'):
        self.args = args
        self.checkpoint_path = checkpoint_path
        if not os.path.exists(self.checkpoint_path):
            self.checkpoint_path.mkdir(parents=True)
        self.recent_epoch = None
        self.filename = filename

    def log(self, *args, **kwargs):
        epoch = kwargs['epoch']

        if self.recent_epoch != epoch:
            self.recent_epoch = epoch
            state_dict = kwargs['state_dict']
            state_dict['epoch'] = kwargs['epoch']
            save_state_dict(state_dict, self.checkpoint_path, self.filename)

    def complete(self, *args, **kwargs):
        save_state_dict(kwargs['state_dict'],
                        self.checkpoint_path, self.filename + '.final')


class BestModelLogger(AbstractBaseLogger):
    def __init__(self, args, checkpoint_path, metric_key, filename='best_acc_model.pth'):
        self.args = args
        self.checkpoint_path = checkpoint_path
        if not os.path.exists(self.checkpoint_path):
            self.checkpoint_path.mkdir(parents=True)

        self.best_metric = 0.
        self.metric_key = metric_key
        self.filename = filename
        self.patience_counter = 0

    def log(self, *args, **kwargs):
        current_metric = kwargs[self.metric_key]
        if self.best_metric < current_metric:  # assumes the higher the better
            print("Update Best {} Model at {}".format(
                self.metric_key, kwargs['epoch']))
            self.best_metric = current_metric
            save_state_dict(kwargs['state_dict'],
                            self.checkpoint_path, self.filename)
            if self.args.early_stopping:
                self.patience_counter = 0
        elif self.args.early_stopping:
            self.patience_counter += 1