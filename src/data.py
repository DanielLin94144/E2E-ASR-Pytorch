import torch
import numpy as np
from functools import partial
from src.text import load_text_encoder
from src.audio import create_transform, ReadAudio, SAMPLE_RATE
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from os.path import join
from src.collect_batch import collect_audio_batch, collect_text_batch, collect_wav_batch
from src.Babeldataset import get_loader

def create_dataset(tokenizer, ascending, text_mode, name, path, bucketing, batch_size, 
                   train_split=None, dev_split=None, test_split=None, read_audio=False, subset=None):
    ''' Interface for creating all kinds of dataset'''

    # Recognize corpus
    if name.lower() == 'librispeech':
        from corpus.preprocess_librispeech import LibriDataset as Dataset
    elif name.lower() == 'dlhlp':
        from corpus.preprocess_dlhlp import DLHLPDataset as Dataset
    else:
        raise NotImplementedError

    # Create dataset
    if train_split is not None:
        # Training mode
        mode = 'train'
        tr_loader_bs = 1 if bucketing and (not ascending) else batch_size
        bucket_size = batch_size if bucketing and (not ascending) else 1 # Ascending without bucketing
        
        if type(dev_split[0]) is not list:
            dv_set = Dataset(path,dev_split,tokenizer, text_mode, 1, read_audio=read_audio, subset=subset) # Do not use bucketing for dev set
            dv_len = len(dv_set)
        else:
            dv_set = []
            for ds in dev_split:
                dev_dir = ''
                if ds[0].lower() == 'librispeech':
                    dev_dir = join(path, 'LibriSpeech')
                    from corpus.preprocess_librispeech import LibriDataset as DevDataset
                else:
                    raise NotImplementedError(ds[0])
                dv_set.append(DevDataset(dev_dir,ds,tokenizer, 1, subset=subset))
            dv_len = sum([len(s) for s in dv_set])
        
        if path[-4:].lower() != name[-4:].lower():
            tr_dir = join(path, name)
        else:
            tr_dir = path
        
        tr_set = Dataset(tr_dir,train_split,tokenizer,text_mode, bucket_size, 
                    ascending=ascending, 
                    read_audio=read_audio,
                    subset=subset)
        # Messages to show
        msg_list = _data_msg(name,path,train_split.__str__(),len(tr_set),
                             dev_split.__str__(),dv_len,batch_size,bucketing)

        return tr_set, dv_set, tr_loader_bs, batch_size, mode, msg_list
    else:
        # Testing model
        mode = 'eval'
        if path[-4:].lower() != name[-4:].lower():
            tt_dir = join(path, name)
        else:
            tt_dir = path
        
        bucket_size = 1
        if type(dev_split[0]) is list: dev_split = dev_split[0]
        
        dv_set = Dataset(tt_dir,dev_split,tokenizer, text_mode, bucket_size, read_audio=read_audio, subset=subset) # Do not use bucketing for dev set
        tt_set = Dataset(tt_dir,test_split,tokenizer, text_mode, bucket_size, read_audio=read_audio, subset=subset) # Do not use bucketing for test set
        # Messages to show
        msg_list = _data_msg(name,tt_dir,dev_split.__str__(),len(dv_set),
                             test_split.__str__(),len(tt_set),batch_size,False)
        msg_list = [m.replace('Dev','Test').replace('Train','Dev') for m in msg_list]
        return dv_set, tt_set, batch_size, batch_size, mode, msg_list

def create_textset(tokenizer, text_mode, train_split, dev_split, name, path, bucketing, batch_size):
    ''' Interface for creating all kinds of text dataset'''
    msg_list = []

    # Recognize corpus
    if name.lower() == "librispeech":
        from corpus.preprocess_librispeech import LibriTextDataset as Dataset
    elif name.lower() == 'dlhlp':
        from corpus.preprocess_dlhlp import DLHLPTextDataset as Dataset
    else:
        raise NotImplementedError

    # Create dataset
    bucket_size = batch_size if bucketing else 1
    tr_loader_bs = 1 if bucketing else batch_size
    dv_set = Dataset(path,dev_split,tokenizer, 1, text_mode) # Do not use bucketing for dev set
    tr_set = Dataset(path,train_split,tokenizer, bucket_size, text_mode)
    
    # Messages to show
    msg_list = _data_msg(name,path,train_split.__str__(),len(tr_set),
                         dev_split.__str__(),len(dv_set),batch_size,bucketing)

    return tr_set, dv_set, tr_loader_bs, batch_size, msg_list


def load_dataset(n_jobs, use_gpu, pin_memory, ascending, corpus, audio, text):
    ''' Prepare dataloader for training/validation'''
    # Text tokenizer
    tokenizer = load_text_encoder(**text)
    # Dataset (in testing mode, tr_set=dv_set, dv_set=tt_set)
    tr_set, dv_set, tr_loader_bs, dv_loader_bs, mode, data_msg = create_dataset(tokenizer,ascending, text['mode'], **corpus)
    # If mode == 'train', tr_set is the train set, dv_set is the development set
    # If mode == 'eval', tr_set is the development set, dv_set is the test set

    # Audio feature extractor
    '''convert to mel-spectrogram'''
    audio_transform_tr, feat_dim = create_transform(audio.copy(), mode=mode)
    audio_transform_dv, feat_dim = create_transform(audio.copy(), mode='eval')

    # Collect function
    collect_tr = partial(collect_audio_batch, audio_transform=audio_transform_tr, mode=mode)
    collect_dv = partial(collect_audio_batch, audio_transform=audio_transform_dv, mode='eval')
    
    # Shuffle/drop applied to training set only
    shuffle = (mode=='train' and not ascending)
    drop_last = shuffle
    # Create data loader

    tr_set = DataLoader(tr_set, batch_size=tr_loader_bs, shuffle=shuffle, drop_last=drop_last, collate_fn=collect_tr,
                        num_workers=n_jobs, pin_memory=use_gpu)
    
    if type(dv_set) is list:
        _tmp_set = []
        for ds in dv_set:
            _tmp_set.append(DataLoader(ds, batch_size=dv_loader_bs, shuffle=False, drop_last=False, collate_fn=collect_dv,
                        num_workers=n_jobs, pin_memory=pin_memory))
        dv_set = _tmp_set
    else:
        dv_set = DataLoader(dv_set, batch_size=dv_loader_bs, shuffle=False, drop_last=False, collate_fn=collect_dv,
                        num_workers=n_jobs, pin_memory=pin_memory)
    
    # Messages to show
    data_msg.append('I/O spec.  | Audio Feature = {}\t| Feature Dim = {}\t| Token Type = {}\t| Vocab Size = {}'\
                    .format(audio['feat_type'],feat_dim,tokenizer.token_type,tokenizer.vocab_size))
    return tr_set, dv_set, feat_dim, tokenizer.vocab_size, tokenizer, data_msg


def load_wav_dataset(n_jobs, use_gpu, pin_memory, ascending, corpus, audio, text):
    # Text tokenizer
    tokenizer = load_text_encoder(**text)
    # Dataset (in testing mode, tr_set=dv_set, dv_set=tt_set)
    tr_set, dv_set, tr_loader_bs, dv_loader_bs, mode, data_msg = create_dataset(tokenizer,ascending,text['mode'], **corpus)
    # If mode == 'train', tr_set is the train set, dv_set is the development set
    # If mode == 'eval', tr_set is the development set, dv_set is the test set
    
    # Audio reader
    tr_audio_reader = ReadAudio(SAMPLE_RATE, mode=mode, time_aug=audio['time_aug'])
    dv_audio_reader = ReadAudio(SAMPLE_RATE, mode='eval', time_aug=audio['time_aug'])
    
    # Collect function
    collect_tr = partial(collect_wav_batch, audio_reader=tr_audio_reader, mode=mode)
    collect_dv = partial(collect_wav_batch, audio_reader=dv_audio_reader, mode='eval')
    
    # Shuffle/drop applied to training set only
    shuffle = (mode=='train' and not ascending)
    drop_last = shuffle
    # Create data loader

    tr_set = DataLoader(tr_set, batch_size=tr_loader_bs, shuffle=shuffle, drop_last=drop_last, collate_fn=collect_tr,
                        num_workers=n_jobs, pin_memory=use_gpu)
    
    if type(dv_set) is list:
        _tmp_set = []
        for ds in dv_set:
            _tmp_set.append(DataLoader(ds, batch_size=dv_loader_bs, shuffle=False, drop_last=False, collate_fn=collect_dv,
                        num_workers=n_jobs, pin_memory=pin_memory))
        dv_set = _tmp_set
    else:
        dv_set = DataLoader(dv_set, batch_size=dv_loader_bs, shuffle=False, drop_last=False, collate_fn=collect_dv,
                        num_workers=n_jobs, pin_memory=pin_memory)
    
    return tr_set, dv_set, tokenizer.vocab_size, tokenizer, data_msg

def load_textset(n_jobs, use_gpu, pin_memory, corpus, text):
    # Text tokenizer
    tokenizer = load_text_encoder(**text)
    # Dataset
    tr_set, dv_set, tr_loader_bs, dv_loader_bs, data_msg = create_textset(tokenizer,text['mode'],**corpus)
    collect_tr = partial(collect_text_batch,mode='train')
    collect_dv = partial(collect_text_batch,mode='eval')
    # Dataloader (Text data stored in RAM, no need num_workers)
    tr_set = DataLoader(tr_set, batch_size=tr_loader_bs, shuffle=True, drop_last=True, collate_fn=collect_tr,
                        num_workers=0, pin_memory=use_gpu)
    dv_set = DataLoader(dv_set, batch_size=dv_loader_bs, shuffle=False, drop_last=False, collate_fn=collect_dv,
                        num_workers=0, pin_memory=pin_memory)

    # Messages to show
    data_msg.append('I/O spec.  | Token type = {}\t| Vocab size = {}'\
                    .format(tokenizer.token_type,tokenizer.vocab_size))

    return tr_set, dv_set, tokenizer.vocab_size, tokenizer, data_msg

"support for babel dataset"
def load_babel_dataset(n_jobs, use_gpu, pin_memory, ascending, max_T, corpus, audio, text):
    # Text tokenizer
    tokenizer = load_text_encoder(**text)

    tr_data_dir = corpus['path']+corpus['train_split'][0]+'/'
    dv_data_dir = corpus['path']+corpus['dev_split'][0]+'/'

    tr_set, tr_dataset_len, tr_max_T = get_loader(tr_data_dir, is_bucket = corpus['bucketing'], batch_size=corpus['batch_size'], \
                                                is_memmap=True, num_workers=n_jobs, max_T = max_T)
    dv_set, dv_dataset_len, dv_max_T = get_loader(dv_data_dir, is_bucket = False, batch_size=corpus['batch_size'], \
                                                is_memmap=True, num_workers=n_jobs, shuffle=False, drop_last=False,  max_T = max_T)

    data_msg = _data_msg(corpus['name'], corpus['path'], corpus['train_split'].__str__(),tr_dataset_len,
                             corpus['dev_split'].__str__(), dv_dataset_len, corpus['batch_size'], corpus['bucketing'])
    
    return tr_set, dv_set, tokenizer.vocab_size, tokenizer, data_msg



def _data_msg(name,path,train_split,tr_set,dev_split,dv_set,batch_size,bucketing):
    ''' List msg for verbose function '''
    msg_list = []
    msg_list.append('Data spec. | Corpus = {} (from {})'.format(name,path))
    msg_list.append('           | Train sets = {}\t| Number of utts = {}'.format(train_split,tr_set))
    msg_list.append('           | Dev sets = {}\t| Number of utts = {}'.format(dev_split,dv_set))
    msg_list.append('           | Batch size = {}\t\t| Bucketing = {}'.format(batch_size,bucketing))
    return msg_list
