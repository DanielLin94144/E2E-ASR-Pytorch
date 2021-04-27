import torch.nn.functional as F
import torch.nn as nn
import torch

from src.util import mask_with_length, get_mask_mean

class _TrainableAugmentModelWithInput(nn.Module):
    def __init__(self, feature_dim, concat_window=11, dim=[256, 256], use_bn=False, init_bias=3., 
                 replace_with_zero=False, init_temp=0.9, max_temp=0.5, max_step=80000, share_random=True, using_hard=False, **kwargs):
        '''
        rand_number_dim: the input rand_number to the generation network
        '''
        super(_TrainableAugmentModelWithInput, self).__init__()
        self.replace_with_zero = replace_with_zero
        self.max_step = max_step
        self.max_temp = max_temp
        self.init_temp = init_temp
        self.slope = (self.max_temp-self.init_temp)/self.max_step

        self.share_random = share_random
        self.using_hard = using_hard

        self.trainable_aug = True # whether this module trainable, or serve as a normal augmentation module

        self.all_models = []  # aug_param layout [T_mask_center, T_mask_width, F_mask_center, F_mask_width] 
        # init T model
        self.all_models.append(_MaskProbB4SigmoidModule(feature_dim, concat_window=concat_window, dim=dim, use_bn=use_bn, init_bias=init_bias))

        self.all_models = nn.ModuleList(self.all_models)

    def set_trainable_aug(self):
        self.trainable_aug = True
        self.train()

    def disable_trainable_aug(self):
        self.trainable_aug = False
        self.eval()

    def forward(self, feat, feat_len, rand_number=None):
        filling_value = 0. if self.replace_with_zero else get_mask_mean(feat, feat_len)
        # print('filling_value', filling_value)
        pre_sigmoid_prob = self._generate_aug_param(feat, feat_len)
        if rand_number is None:
            rand_number = self.get_new_rand_number(feat)
        # print('aug_param', aug_param)

        if self.trainable_aug:
            return self._forward_trainable(feat, feat_len, filling_value, pre_sigmoid_prob, rand_number)
        else:
            return self._forward_not_trainable(feat, feat_len, filling_value, pre_sigmoid_prob, rand_number)

    def get_new_rand_number(self, feat):
        if self.share_random:
            U = torch.rand_like(feat[:, 0, 0])
        else:
            U = torch.rand_like(feat[:, :, 0])
        return U

    def _generate_aug_param(self, feat, feat_len):
        '''
        std: None or [T_center_input_std., T_center_input_std, F_center_input_std., F_center_input_std]
        '''
        return self.all_models[0](feat, feat_len) # [B, 2*(T_num_masks+F_num_masks)]

    def set_step(self, step): 
        self.step = step

    def temp_scheduler(self):
        assert (self.step is not None)
        curr_temp = self.init_temp + self.step * self.slope

        return curr_temp
        
    def _gumbel(self, pre_sigmoid_prob, rand_number, temp=0.9, eps=1e-20):
        # https://arxiv.org/pdf/1806.02988.pdf
        U = rand_number
        if self.share_random:
            sample_gumbel = torch.log(U + eps) - torch.log(1 - U + eps)
            sample_gumbel = sample_gumbel.unsqueeze(-1).expand_as(pre_sigmoid_prob)
        else:
            sample_gumbel = torch.log(U + eps) - torch.log(1 - U + eps)

        gumbel_output = pre_sigmoid_prob + sample_gumbel
        soft_prob = F.sigmoid(gumbel_output / temp)

        hard_prob = torch.where(soft_prob>0.5, torch.ones_like(pre_sigmoid_prob), torch.zeros_like(pre_sigmoid_prob))
        hard_prob = (hard_prob - soft_prob).detach() + soft_prob
        return soft_prob, hard_prob

    def _forward_trainable(self, feat, feat_len, filling_value, pre_sigmoid_prob, rand_number):
        '''
        filling_value: [B]
        aug_param: [B, 2*(F_num_mask+T_num_mask)]
        '''
        curr_temp = self.temp_scheduler()
        soft_prob, hard_prob = self._gumbel(pre_sigmoid_prob, rand_number, temp=curr_temp)
        if self.using_hard:
            total_left_weight = hard_prob.unsqueeze(-1)
        else:
            total_left_weight = soft_prob.unsqueeze(-1)
        new_feat = feat*total_left_weight + filling_value.unsqueeze(1).unsqueeze(1)*(1-total_left_weight)
        # mask with feat_len
        new_feat, _ = mask_with_length(new_feat, feat_len)
        return new_feat    

    def _forward_not_trainable(self, feat, feat_len, filling_value, pre_sigmoid_prob, rand_number):
        _, hard_prob = self._gumbel(pre_sigmoid_prob, rand_number)
        total_left_weight = hard_prob.unsqueeze(-1)
        new_feat = feat*total_left_weight + filling_value.unsqueeze(1).unsqueeze(1)*(1-total_left_weight)
        # mask with feat_len
        new_feat, _ = mask_with_length(new_feat, feat_len)
        return new_feat    
    
    def log_info(self):
        print(f'[INFO] - temperature = {self.temp_scheduler()} @ step {self.step}')

class _MaskProbB4SigmoidModule(nn.Module):
    def __init__(self, feature_dim, concat_window=5, dim=[256, 256], use_bn=False, init_bias=-3.):
        super(_MaskProbB4SigmoidModule, self).__init__()
        assert concat_window%2==1
        self.concat_window = concat_window
        self.model = self.create_model(feature_dim, concat_window, dim, use_bn, init_bias)

    def create_model(self, feature_dim, concat_window, dim, use_bn, init_bias):
        module_list = []
        prev_dim = feature_dim*concat_window
        for d in dim:
            linear = nn.Linear(prev_dim, d)
            nn.init.kaiming_normal_(linear.weight, nonlinearity='relu')
            module_list.append(linear)
            prev_dim = d
            module_list.append(nn.ReLU())
            if use_bn:
                module_list.append(nn.BatchNorm1d(d))
            
        module_list.append(nn.Linear(prev_dim, 1))
        module_list[-1].bias.data.fill_(init_bias)

        return nn.Sequential(*module_list)

    def forward(self, feat, feat_len):
        # padding
        feat, _ = mask_with_length(feat, feat_len)
        # expand
        half_window = self.concat_window // 2
        #_feat_ = torch.cat([feat.zeros_like(B, half_window, d), feat, feat.zeros_like(B, half_window, d)], dim=1)
        expand_feat = F.unfold(feat.permute(0, 2, 1).unsqueeze(-1), (self.concat_window, 1), padding=(half_window, 0)).permute(0, 2, 1)
        # generate log prob
        return self.model(expand_feat).squeeze(-1)

if __name__ == '__main__':
    batch_size = 3
    l = 11
    feat_dim = 5
    feat = torch.rand(batch_size, l, feat_dim).cuda()
    # feat_len = torch.randint(1, l-1, size=(batch_size,)).cuda()
    feat_len = torch.tensor([11,9,7]).cuda()
    feat, _ = mask_with_length(feat, feat_len)

    module = _MaskProbB4SigmoidModule(feature_dim=feat_dim).cuda()
    result = module(feat, feat_len)
    print(result.shape)

    # testing normal case
    trainable_aug = _TrainableAugmentModelWithInput(feature_dim=feat_dim, share_random=False, concat_window=3, init_bias=2.).cuda()
    trainable_aug.set_step(0)
    print(feat)
    for i in range(1):
        print('test soft')
        trainable_aug.set_trainable_aug()
        new_feat_soft  = trainable_aug(feat, feat_len)
        print(new_feat_soft.shape)
        print(new_feat_soft-feat)
        print('test hard')
        trainable_aug.disable_trainable_aug()
        new_feat_hard  = trainable_aug(feat, feat_len)
        print(new_feat_hard.shape)
        print(new_feat_hard-feat)


