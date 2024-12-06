import os
import time
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn


def compute_confidence_interval(data):
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm


def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cuda_deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        cudnn.deterministic = False
        cudnn.benchmark = True


class Logger(object):
    def __init__(self, path, mode='train'):
        self.birth_time = time.strftime('%Yy_%mm_%dd_%Hh_%Mm_%Ss', time.localtime())
        self.log_path = os.path.join(path + f'/{mode}', f'{self.birth_time}.log')
        self.args_path = os.path.join(path, f'args.log')
        self.parameters_path = os.path.join(path, 'parameters.log')
        if mode == 'train':
            self.prefix = 'Train'
        elif mode == 'val':
            self.prefix = 'Val'
        elif mode == 'test':
            self.prefix = 'Test'
        else:
            self.prefix = 'Log'

    def log(self, info, prefix=None, verbose=True):
        """ set verbose=False if you don't want the log info be print """
        with open(self.log_path, 'a+') as f:
            time_stamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            f.write(time_stamp + '\t' + info + '\n')
        if verbose:
            if prefix:
                print(f'[{prefix}] ', info)
            else:
                print(f'[{self.prefix}] ', info)

    def log_args(self, args):
        with open(self.args_path, 'w+') as f:
            f.write('\n'.join([f'{k}: {v}' for k, v in vars(args).items()]))

    def log_parameters(self, model):
        log_parameters = []
        for name, p in model.named_parameters():
            if p.requires_grad:
                log_parameters.append(name)
        with open(self.parameters_path, 'w+') as f:
            f.write('\n'.join(log_parameters))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':4f'):
        self.name = name
        self.fmt = fmt
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
