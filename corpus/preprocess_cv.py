import argparse
import os
import torchaudio
from copy import deepcopy
import torch
import time
import random
import math
import json
import subprocess
import sys
import progressbar
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.multiprocessing import Pool


def load(path_item):
    seq_name = path_item.stem
    data = torchaudio.load(str(path_item))[0].view(1, -1)
    return seq_name, data

def findAllSeqs(dirName,
                extension='.flac',
                loadCache=False,
                speaker_level=1):
    r"""
    Lists all the sequences with the given extension in the dirName directory.
    Output:
        outSequences, speakers
        outSequence
        A list of tuples seq_path, speaker where:
            - seq_path is the relative path of each sequence relative to the
            parent directory
            - speaker is the corresponding speaker index
        outSpeakers
        The speaker labels (in order)
    The speaker labels are organized the following way
    \dirName
        \speaker_label
            \..
                ...
                seqName.extension
    Adjust the value of speaker_level if you want to choose which level of
    directory defines the speaker label. Ex if speaker_level == 2 then the
    dataset should be organized in the following fashion
    \dirName
        \crappy_label
            \speaker_label
                \..
                    ...
                    seqName.extension
    Set speaker_label == 0 if no speaker label will be retrieved no matter the
    organization of the dataset.
    """
    cache_path = os.path.join(dirName, '_seqs_cache.txt')
    if loadCache:
        try:
            outSequences, speakers = torch.load(cache_path)
            print(f'Loaded from cache {cache_path} successfully')
            return outSequences, speakers
        except OSError as err:
            print(f'Ran in an error while loading {cache_path}: {err}')
        print('Could not load cache, rebuilding')

    if dirName[-1] != os.sep:
        dirName += os.sep
    prefixSize = len(dirName)
    speakersTarget = {}
    outSequences = []
    for root, dirs, filenames in tqdm.tqdm(os.walk(dirName)):
        filtered_files = [f for f in filenames if f.endswith(extension)]

        if len(filtered_files) > 0:
            speakerStr = (os.sep).join(
                root[prefixSize:].split(os.sep)[:speaker_level])
            if speakerStr not in speakersTarget:
                speakersTarget[speakerStr] = len(speakersTarget)
            speaker = speakersTarget[speakerStr]
            for filename in filtered_files:
                full_path = os.path.join(root[prefixSize:], filename)
                outSequences.append((speaker, full_path))
    outSpeakers = [None for x in speakersTarget]
    for key, index in speakersTarget.items():
        outSpeakers[index] = key
    try:
        torch.save((outSequences, outSpeakers), cache_path)
        print(f'Saved cache file at {cache_path}')
    except OSError as err:
        print(f'Ran in an error while saving {cache_path}: {err}')
    return outSequences, outSpeakers

def parseSeqLabels(pathLabels):
    with open(pathLabels, 'r') as f:
        lines = f.readlines()
    output = {"step": 160}  # Step in librispeech dataset is 160bits
    maxPhone = 0
    for line in lines:
        data = line.split()
        output[data[0]] = [int(x) for x in data[1:]]
        maxPhone = max(maxPhone, max(output[data[0]]))
    return output, maxPhone + 1

def filterSeqs(pathTxt, seqCouples):
    with open(pathTxt, 'r') as f:
        inSeqs = [p.replace('\n', '') for p in f.readlines()]

    inSeqs.sort()
    # print(inSeqs)
    seqCouples.sort(key=lambda x: os.path.basename(os.path.splitext(x[1])[0]))
    output, index = [], 0
    for x in seqCouples:
        seq = os.path.basename(os.path.splitext(x[1])[0])
        while index < len(inSeqs) and seq > inSeqs[index]:
            index += 1
        if index == len(inSeqs):
            break
        if seq == inSeqs[index]:
            output.append(x)
    return output

class SingleSequenceDataset(Dataset):

    def __init__(self,
                 pathDB,
                 seqNames,
                 phoneLabelsDict,
                 inDim=1,
                 transpose=True):
        """
        Args:
            - path (string): path to the training dataset
            - sizeWindow (int): size of the sliding window
            - seqNames (list): sequences to load
            - phoneLabels (dictionnary): if not None, a dictionnary with the
                                         following entries
                                         "step": size of a labelled window
                                         "$SEQ_NAME": list of phonem labels for
                                         the sequence $SEQ_NAME
        """
        self.seqNames = deepcopy(seqNames)
        self.pathDB = pathDB
        self.phoneLabelsDict = deepcopy(phoneLabelsDict)
        self.inDim = inDim
        self.transpose = transpose
        self.loadSeqs()

    def loadSeqs(self):

        # Labels
        self.seqOffset = [0]
        self.phoneLabels = []
        self.phoneOffsets = [0]
        self.data = []
        self.maxSize = 0
        self.maxSizePhone = 0

        # Data

        nprocess = min(30, len(self.seqNames))

        start_time = time.time()
        to_load = [Path(self.pathDB) / x for _, x in self.seqNames]

        with Pool(nprocess) as p:
            poolData = p.map(load, to_load)

        tmpData = []
        poolData.sort()

        totSize = 0
        minSizePhone = float('inf')
        for seqName, seq in poolData:
            self.phoneLabels += self.phoneLabelsDict[seqName]
            self.phoneOffsets.append(len(self.phoneLabels))
            self.maxSizePhone = max(self.maxSizePhone, len(
                self.phoneLabelsDict[seqName]))
            minSizePhone = min(minSizePhone, len(
                self.phoneLabelsDict[seqName]))
            sizeSeq = seq.size(1)
            self.maxSize = max(self.maxSize, sizeSeq)
            totSize += sizeSeq
            tmpData.append(seq)
            self.seqOffset.append(self.seqOffset[-1] + sizeSeq)
            del seq
        self.data = torch.cat(tmpData, dim=1)
        self.phoneLabels = torch.tensor(self.phoneLabels, dtype=torch.long)
        print(f'Loaded {len(self.phoneOffsets)} sequences '
              f'in {time.time() - start_time:.2f} seconds')
        print(f'maxSizeSeq : {self.maxSize}')
        print(f'maxSizePhone : {self.maxSizePhone}')
        print(f"minSizePhone : {minSizePhone}")
        print(f'Total size dataset {totSize / (16000 * 3600)} hours')

    def __getitem__(self, idx):

        offsetStart = self.seqOffset[idx]
        offsetEnd = self.seqOffset[idx+1]
        offsetPhoneStart = self.phoneOffsets[idx]
        offsetPhoneEnd = self.phoneOffsets[idx + 1]

        sizeSeq = int(offsetEnd - offsetStart)
        sizePhone = int(offsetPhoneEnd - offsetPhoneStart)

        outSeq = torch.zeros((self.inDim, self.maxSize))
        outPhone = torch.zeros((self.maxSizePhone))

        outSeq[:, :sizeSeq] = self.data[:, offsetStart:offsetEnd]
        outPhone[:sizePhone] = self.phoneLabels[offsetPhoneStart:offsetPhoneEnd]

        return outSeq,  torch.tensor([sizeSeq], dtype=torch.long), outPhone.long(),  torch.tensor([sizePhone], dtype=torch.long)

    def __len__(self):
        return len(self.seqOffset) - 1

if __name__ == '__main__':

    well_lang = ['es']
    for lang in well_lang:
        pathDB = '/home/daniel094144/Daniel/data/CommonVoice/'+lang+'/clips_16k'    
        pathPhone = '/home/daniel094144/Daniel/data/CommonVoice/'+lang+'/reduced_global_IPA.txt'
        file_extension = '.mp3'
        in_dim = 1
        pathTrain = '/home/daniel094144/Daniel/common_voices_splits/'+lang+'/trainSeqs_5.0_uniform_new_version.txt'
        batchSize = 4
        '''
        variable setting

            - pathDB: Path to the directory containing the audio data / pre-computed features.
            - PathPhone: Path to the .txt file containing the phone transcription.
            - in_dim: Dimension of the input data, useful when working with 
                            pre-computed features or stereo audio.
            - pathTrain: Path to the .txt files containing the list of the training sequences.                       
        '''

        inSeqs, _ = findAllSeqs(pathDB,
                                extension=file_extension)
        
        phoneLabels, nPhones = parseSeqLabels(pathPhone)

        seqTrain = filterSeqs(pathTrain, inSeqs)
        # print(seqTrain)
        print(f"Loading the training dataset at {pathDB}")

        datasetTrain = SingleSequenceDataset(pathDB, seqTrain,
                                                phoneLabels, inDim=in_dim)

        train_loader = DataLoader(datasetTrain, batch_size=batchSize,
                                    shuffle=True)

        for data in train_loader:
            seq, sizeSeq, phone, sizePhone = data
            print(seq.shape)
            print(sizeSeq)
            print(phone)
            print(sizePhone)