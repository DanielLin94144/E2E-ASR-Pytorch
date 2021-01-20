import torch
import torch.nn as nn
import torch.nn.functional as F

class TrainableAugment(nn.Module):
    def __init__(self, T_num_masks=1, F_num_masks=1, noise_dim=None, dim=[10, 10, 10], replace_with_zero=False, width_init_bias=-3.):
        '''
        noise_dim: the input noise to the generation network
        '''
        super(TrainableAugment, self).__init__()
        self.T_num_masks = T_num_masks
        self.F_num_masks = F_num_masks

        self.output_dim = (self.T_num_masks+self.F_num_masks)*2 # position and width
        if noise_dim is None:
            self.noise_dim = self.output_dim
        else:
            self.noise_dim = noise_dim

        self.dim = dim
        self.replace_with_zero = replace_with_zero

        self.trainable_aug = True # whether this module trainable, or serve as a normal augmentation module

        module_list = []
        prev_dim = self.noise_dim
        for d in self.dim:
            module_list.append(nn.Linear(prev_dim, d))
            prev_dim = d
            module_list.append(nn.ReLU())
        module_list.append(nn.Linear(prev_dim, self.output_dim))
        module_list.append(nn.Sigmoid())

        # init last linear 
        # aug_param layout [T_mask_center, T_mask_width, F_mask_center, F_mask_width] 
        # init mask width to be small 
        module_list[-2].bias[self.T_num_masks:2*self.T_num_masks] = width_init_bias
        module_list[-2].bias[(2*self.T_num_masks+self.F_num_masks):(2*self.T_num_masks+2*self.F_num_masks)] = width_init_bias
        self.layers = nn.Sequential(*module_list)

    def set_trainable_aug(self):
        self.trainable_aug = True

    def disable_trainable_aug(self):
        self.trainable_aug = False

    def forward(self, feat, feat_len):
        filling_value = 0. if self.replace_with_zero else self._get_mask_mean(feat, feat_len)
        print('filling_value', filling_value)
        aug_param = self._generate_aug_param(feat)
        print('aug_param', aug_param)

        if self.trainable_aug:
            return self._forward_trainable(feat, feat_len, filling_value, aug_param)
        else:
            return self._forward_not_trainable(feat, feat_len, filling_value, aug_param)
    
    def _generate_aug_param(self, feat):
        noise = torch.randn(feat.shape[0], self.noise_dim).to(feat.device)
        aug_param = self.layers(noise)
        return aug_param # [B, 2*(T_num_masks+F_num_masks)]

    def _forward_trainable(self, feat, feat_len, filling_value, aug_param):
        '''
        filling_value: [B]
        aug_param: [B, 2*(F_num_mask+T_num_mask)]
        '''
        def _get_soft_log_left_weight(mask_center, mask_width, length, max_len):
            '''
            length: [B] # can be T or F dimension
            mask_center: [B, num_mask]
            mask_width: [B, num_mask]
            output: [B, l]
            '''
            SIGMOID_THRESHOLD = 5 # assume 2*SIGMOID_THRESHOLD is the width, because sigmoid(SIGMOID_THRESHOLD) starts to very close to 1
            device = mask_center.device

            position = torch.div(torch.range(start=0, end=max_len-1, device=device).unsqueeze(0), length.unsqueeze(1)) # [B, l]

            dist_to_center = mask_center.unsqueeze(-1) - position.unsqueeze(1) # [B, num_mask, l]
            abs_ratio = torch.abs(dist_to_center)/mask_width.unsqueeze(-1) # [B, num_mask, l]

            log_mask_weight = F.logsigmoid((abs_ratio*(2*SIGMOID_THRESHOLD))-SIGMOID_THRESHOLD) # [B, num_mask, l]
            log_mask_weight = log_mask_weight.sum(1) # [B, l]
            return log_mask_weight

        # mask T
        T_log_mask_weight = _get_soft_log_left_weight(aug_param[:, :self.T_num_masks], \
                                                      aug_param[:, self.T_num_masks:2*self.T_num_masks], \
                                                      feat_len, feat.shape[1])
        print('T mask', T_log_mask_weight)
        # mask F
        F_log_mask_weight = _get_soft_log_left_weight(aug_param[:, 2*self.T_num_masks:2*self.T_num_masks+self.F_num_masks], \
                                                      aug_param[:, 2*self.T_num_masks+self.F_num_masks:2*self.T_num_masks+2*self.F_num_masks], \
                                                      feat_len.new_tensor([feat.shape[-1]]), feat.shape[-1])
        print('F mask', F_log_mask_weight)
        total_log_left_weight = T_log_mask_weight.unsqueeze(-1)+F_log_mask_weight.unsqueeze(1) # [B, T, F]
        total_left_weight = torch.exp(total_log_left_weight)

        new_feat = feat*total_left_weight + filling_value.unsqueeze(1).unsqueeze(1)*(1-total_left_weight)
        # we do not mask the fill result, cuz assume ASR model will handle this
        return new_feat

    def _forward_not_trainable(self, feat, feat_len, filling_value, aug_param):
        # use fix 
        def _get_hard_mask(mask_center, mask_width, length, max_len):
            '''
            length: [B] # can be T or F dimension
            mask_center: [B, num_mask]
            mask_width: [B, num_mask]
            output: [B, l]
            '''
            device = mask_center.device
            position = torch.div(torch.range(start=0, end=max_len-1, device=device).unsqueeze(0), length.unsqueeze(1)) # [B, l]

            mask_left = mask_center  - mask_width/2 - torch.rand_like(mask_center)/length.unsqueeze(-1)
            mask_right = mask_center + mask_width/2 + torch.rand_like(mask_center)/length.unsqueeze(-1)

            mask_left  = position.unsqueeze(1) > mask_left.unsqueeze(-1) # [B, num_mask, l]
            mask_right = position.unsqueeze(1) < mask_right.unsqueeze(-1)
            mask = mask_left & mask_right
            mask = mask.any(1)

            return mask

        # mask T
        T_mask = _get_hard_mask(aug_param[:, :self.T_num_masks], \
                                aug_param[:, self.T_num_masks:2*self.T_num_masks], \
                                feat_len, feat.shape[1])
        # mask F
        F_mask = _get_hard_mask(aug_param[:, 2*self.T_num_masks:2*self.T_num_masks+self.F_num_masks], \
                                aug_param[:, 2*self.T_num_masks+self.F_num_masks:2*self.T_num_masks+2*self.F_num_masks], \
                                feat_len.new_tensor([feat.shape[-1]]), feat.shape[-1])

        total_mask = T_mask.unsqueeze(-1) | F_mask.unsqueeze(1)

        new_feat = torch.where(total_mask, filling_value.unsqueeze(1).unsqueeze(1), feat)
        # we do not mask the fill result, cuz assume ASR model will handle this
        return new_feat    

    def _mask_with_length(self, feat, feat_len, fill_num=0.):
        '''
        mask the second dimensin of the feat with feat_len
        feat: rank>=2 [B x L x ...]
        feat_len: [B]
        '''
        assert (len(feat.size()) >= 2)
        assert (feat.size()[0] == feat_len.size()[0])

        max_len = feat.size()[1]
        rank = len(feat.size())

        a = torch.arange(max_len).unsqueeze(0).int()
        b = feat_len.unsqueeze(1).int()
        if feat.is_cuda:
            a = a.cuda()
            b = b.cuda()

        mask = torch.ge(a, b)
        mask = mask.view(mask.size()[0],mask.size()[1], *([1]*(rank-2))).expand_as(feat) #mask: where to pad zero
        #if feat.is_cuda:
            #mask = mask.cuda()
        feat_clone = feat.clone() # prevent inplace substitution
        feat_clone[mask] = fill_num
        return feat_clone, ~mask

    def _get_mask_mean(self, feat, feat_len):
        feat, mask = self._mask_with_length(feat, feat_len)
        return feat.sum([1,2])/mask.float().sum([1,2])

if __name__ == '__main__':
    trainable_aug = TrainableAugment().cuda()

    batch_size = 3
    l = 20
    feat_dim = 5
    feat = torch.rand(batch_size, l, feat_dim).cuda()
    # feat_len = torch.randint(1, l-1, size=(batch_size,)).cuda()
    feat_len = torch.tensor([l]*batch_size).cuda()

    new_feat_soft  = trainable_aug(feat, feat_len)

    trainable_aug.disable_trainable_aug()
    new_feat_hard  = trainable_aug(feat, feat_len)


