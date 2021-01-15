#!/usr/bin/env python
# coding: utf-8
import os
import yaml
import torch
import random
import argparse
import numpy as np


# Arguments
parser = argparse.ArgumentParser(description='Training E2E asr.')
parser.add_argument('--config', type=str, help='Path to experiment config.')
parser.add_argument('--name', default=None, type=str, help='Name for logging.')
parser.add_argument('--logdir', default='log/', type=str, help='Logging path.', required=False)
parser.add_argument('--ckpdir', default='ckpt/', type=str, help='Checkpoint path.', required=False)
parser.add_argument('--outdir', default='result/', type=str, help='Decode output path.', required=False)
parser.add_argument('--load', default=None, type=str, help='Load pre-trained model (for training only)', required=False)
parser.add_argument('--seed', default=0, type=int, help='Random seed for reproducable results.', required=False)
parser.add_argument('--cudnn-ctc', action='store_true', help='Switches CTC backend from torch to cudnn')
parser.add_argument('--njobs', default=4, type=int, help='Number of threads for dataloader/decoding.', required=False)
parser.add_argument('--cpu', action='store_true', help='Disable GPU training.')
parser.add_argument('--no-pin', action='store_true', help='Disable pin-memory for dataloader')
parser.add_argument('--test', action='store_true', help='Test the model.')
parser.add_argument('--no-msg', action='store_true', help='Hide all messages.')
parser.add_argument('--lm', action='store_true', help='Option for training RNNLM.')
parser.add_argument('--amp', action='store_true', help='Option to enable AMP.')
parser.add_argument('--reserve_gpu', default=0, type=float, help='Option to reserve GPU ram for training.')
parser.add_argument('--jit', action='store_true', help='Option for enabling jit in pytorch. (feature in development)')
parser.add_argument('--cuda', default=0, type=int, help='Choose which gpu to use.')
parser.add_argument('--deterministic', action='store_true', help='Ensuring same behavior')
parser.add_argument('--no_cudnn', action='store_true', help='Ensuring same behavior')
parser.add_argument(
    '--upstream',
    help='Specify the upstream variant according to torch.hub.list'
)
parser.add_argument(
    '--upstream_feature_selection',
    help=f'Specify the layer to be extracted as the representation according to torch.hub.help'
)
parser.add_argument(
    '--upstream_refresh',
    action='store_true',
    help='Re-download cached ckpts for on-the-fly upstream variants'
)
parser.add_argument(
    '--upstream_ckpt',
    metavar='{PATH,URL,GOOGLE_DRIVE_ID}',
    help='Only set when the specified upstream has \'ckpt\' as an argument in torch.hub.help'
)
parser.add_argument(
    '--upstream_trainable',
    action='store_true',
    help='Whether to finetune the upstream model'
)

paras = parser.parse_args()
setattr(paras,'gpu',not paras.cpu)
setattr(paras,'pin_memory',not paras.no_pin)
setattr(paras,'verbose',not paras.no_msg)
config = yaml.load(open(paras.config,'r'), Loader=yaml.FullLoader)

print('[INFO] Using config {}'.format(paras.config))

random.seed(paras.seed)
np.random.seed(paras.seed)
torch.manual_seed(paras.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(paras.seed)
    # print('There are ', torch.cuda.device_count(), ' device(s) available')
    # print('Using device cuda:', str(paras.cuda))
    if paras.deterministic:
        print('Set torch.deterministic(True) for reproducible debugging, ctc_weight should be 0')
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.set_deterministic(True)
        torch.backends.cudnn.enabled = False
    elif paras.no_cudnn:
        torch.backends.cudnn.enabled = False
    else:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Hack to preserve GPU ram just incase OOM later on server
if paras.gpu and paras.reserve_gpu>0:
    buff = torch.randn(int(paras.reserve_gpu*1e9//4)).to(torch.device('cuda:' + str(paras.cuda)))
    del buff

if paras.lm:
    # Train RNNLM
    from bin.train_lm import Solver
    mode = 'train'
else:
    if paras.test:
        # Test ASR
        assert paras.load is None, 'Load option is mutually exclusive to --test'
        from bin.test_asr import Solver
        mode = 'test'
    else:
        # Train ASR
        from bin.train_asr import Solver
        mode = 'train'

solver = Solver(config,paras,mode)
solver.load_data()
solver.set_model()
solver.exec()
