# End-to-end Automatic Speech Recognition Systems - PyTorch Implementation
For complete introdution and usage, please see the original repository [Alexander-H-Liu/End-to-end-ASR-Pytorch](https://github.com/Alexander-H-Liu/End-to-end-ASR-Pytorch).
## New features
1. 
## Instructions
### Training
Modify `script/train.sh`, `script/train_lm.sh`, `config/librispeech_asr.yaml`, and `config/librispeech_lm.yaml` first. GPU is required.
```
bash script/train.sh <asr name> <cuda id>
bash script/train_lm.sh <lm name> <cuda id>
```
### Testing
Modify `script/test.sh` and `config/librispeech_test.sh` first. Increase the number of `--njobs` can speed up decoding process, but might cause OOM.
```
bash script/test.sh <asr name> <cuda id>
```

## LibriSpeech 100hr Baseline
This baseline is composed of a character-based joint CTC-attention ASR model and an RNNLM which were trained on the LibriSpeech `train-clean-100`. The perplexity of the LM on the `dev-clean` set is 3.66. 

| Decoding | DEV WER(%) | TEST WER(%) |
| -------- | ---------- | ----------- |
| Greedy   |            |             |

## TODO
1. CTC beam decoding (testing)
2. SpecAugment (will be released)
3. Multiple corpora training (will be released)
4. Support of WSJ and Switchboard dataset (under construction)
5. Combination of CTC and RNNLM: RNN transducer (under construction)

## Citation

```
@inproceedings{liu2019adversarial,
  title={Adversarial Training of End-to-end Speech Recognition Using a Criticizing Language Model},
  author={Liu, Alexander and Lee, Hung-yi and Lee, Lin-shan},
  booktitle={International Conference on Speech RecognitionAcoustics, Speech and Signal Processing (ICASSP)},
  year={2019},
  organization={IEEE}
}

@inproceedings{alex2019sequencetosequence,
    title={Sequence-to-sequence Automatic Speech Recognition with Word Embedding Regularization and Fused Decoding},
    author={Alexander H. Liu and Tzu-Wei Sung and Shun-Po Chuang and Hung-yi Lee and Lin-shan Lee},
    booktitle={International Conference on Speech RecognitionAcoustics, Speech and Signal Processing (ICASSP)},
    year={2020},
    organization={IEEE}
}

@misc{chang2020endtoend,
    title={End-to-end Whispered Speech Recognition with Frequency-weighted Approaches and Layer-wise Transfer Learning},
    author={Heng-Jui Chang and Alexander H. Liu and Hung-yi Lee and Lin-shan Lee},
    year={2020},
    eprint={2005.01972},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
