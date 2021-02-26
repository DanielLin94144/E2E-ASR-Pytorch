import os
import torch 
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    SIGMOID_THRESHOLD = 50

    feat = torch.rand(1, 300, 80)

    '''soft time masking'''
    len_spectro = feat.shape[1]
    t_zero = len_spectro*torch.rand(1,1)
    T = Variable(20*torch.ones(1,1), requires_grad=True)
    ratio = torch.rand(1,1)

    t = T*ratio
    print("t", t)

    position = torch.arange(start=0, end=len_spectro).unsqueeze(0) # [1, l]

    mask_center_recover = t_zero
    mask_width_recover = t

    dist_to_center = mask_center_recover.unsqueeze(-1) - position.unsqueeze(1) # [B, num_mask, l]
    abs_ratio = torch.abs(dist_to_center)/(mask_width_recover.unsqueeze(-1)+1e-8) # [B, num_mask, l]
    t_mask_weight = torch.sigmoid((abs_ratio*(2*SIGMOID_THRESHOLD))-SIGMOID_THRESHOLD) # [B, num_mask, l]
    t_mask_weight = t_mask_weight.sum(1)

    '''soft freq masking'''
    feat_dim = feat.shape[2]
    f_zero = feat_dim*torch.rand(1,1)
    F = Variable(10*torch.ones(1,1), requires_grad=True)
    ratio = torch.rand(1,1)

    f = F*ratio
    print("f", f)

    position = torch.arange(start=0, end=feat_dim).unsqueeze(0) # [1, l]

    mask_center_recover = f_zero
    mask_width_recover = f

    dist_to_center = mask_center_recover.unsqueeze(-1) - position.unsqueeze(1) # [B, num_mask, l]
    abs_ratio = torch.abs(dist_to_center)/(mask_width_recover.unsqueeze(-1)+1e-8) # [B, num_mask, l]
    f_mask_weight = torch.sigmoid((abs_ratio*(2*SIGMOID_THRESHOLD))-SIGMOID_THRESHOLD) # [B, num_mask, l]
    f_mask_weight = f_mask_weight.sum(1)

    print('t mask', t_mask_weight.shape)
    print('f mask', f_mask_weight.shape)

    total_left_weight = t_mask_weight.unsqueeze(-1)*f_mask_weight.unsqueeze(1) # [B, T, F]
    print('total', total_left_weight.shape)

    filling_value = torch.tensor([0.])

    new_feat = feat*total_left_weight + \
            filling_value.unsqueeze(1).unsqueeze(1)*(1-total_left_weight)

    print(feat)
    print(new_feat)


    print(T.grad)
    print(F.grad)
    new_feat.sum().backward()
    print(T.grad) 
    print(F.grad)

    feat = feat.cpu().detach().numpy()
    feat = np.squeeze(feat)
    new_feat = new_feat.cpu().detach().numpy()
    new_feat = np.squeeze(new_feat)
    
    plt.imsave('./feat.png', feat)
    plt.imsave('./new_feat.png', new_feat)