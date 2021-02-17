import torch 
import os
import yaml
import numpy as np
from src.data import load_babel_dataset
from src.augmentation import TrainableAugment
import matplotlib

if __name__ == '__main__':
    yaml_path = '/Home/daniel094144/E2E-ASR-Pytorch/config/babel_202_trainable_aug.yaml'
    ckpt_path = '/Home/daniel094144/E2E-ASR-Pytorch/ckpt/nofast_init/best_aug.pth'

    config = yaml.load(open(yaml_path,'r'), Loader=yaml.FullLoader)

    aug = TrainableAugment(config['augmentation']['type'], \
                                            config['augmentation']['trainable_aug']['model'], \
                                            config['augmentation']['trainable_aug']['optimizer'])


    aug.load_ckpt(ckpt_path)
    aug.aug_model = aug.aug_model.cuda()

    config['data']['corpus']['batch_size'] = 1
    tr_set, dv_set, vocab_size, tokenizer, msg = \
                    load_babel_dataset(0, True, True, 
                                True, **config['data'])
    tr_data = next(iter(tr_set))
    feat, feat_len, txt, txt_len = tr_data
    feat = feat.cuda()
    feat_len = feat_len.cuda()
    print(feat.shape)
    print(feat_len)

    new_feat = aug.aug_model(feat, feat_len)
    feat = feat.cpu().detach().numpy()
    feat = np.squeeze(feat)
    new_feat = new_feat.cpu().detach().numpy()
    new_feat = np.squeeze(new_feat)

    feat = np.transpose(feat)
    new_feat = np.transpose(new_feat)

    os.makedirs('./demo', exist_ok=True)
    matplotlib.image.imsave('./demo/feat.png', feat)
    matplotlib.image.imsave('./demo/new_feat.png', new_feat)


        