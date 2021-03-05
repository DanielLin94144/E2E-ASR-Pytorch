import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class TrainableAugment(nn.Module): 
    def __init__(self, aug_type, model, optimizer, max_T):
        super(TrainableAugment, self).__init__()
        self.aug_type = aug_type # 1: train aug 2: use pretrain aug module 3:previous specaug 4: no aug(only means no aug on spectrogram)
        self.max_T = max_T
        if self.aug_type in [3,4]:
            self.aug_model = None
            self.optimizer = None
        elif self.aug_type == 1:
            self.aug_model = _TrainableAugmentModel(self.max_T, **model)
            self.aug_model.set_trainable_aug()
            opt = getattr(torch.optim, optimizer.pop('optimizer', 'sgd'))
            self.optimizer = opt(self.aug_model.parameters(), **optimizer)
        elif self.aug_type == 2:
            self.aug_model = _TrainableAugmentModel(self.max_T, **model)
            self.aug_model.disable_trainable_aug()
            self.optimizer = None
        else:
            raise NotImplementedError

    def set_step(self, step): 
        if self.aug_model:
            self.aug_model.set_step(step)

    def get_sigmoid_threshold(self):
        if self.aug_model:
            return self.aug_model.sigmoid_threshold_scheduler()
        else:
            return None

    def get_new_noise(self, feat):
        if self.aug_model is None:
            return None
        else:
            return self.aug_model.get_new_noise(feat)

    def forward(self, feat, feat_len, noise=None):
        if self.aug_type == 1 or self.aug_type == 2:
            feat = self.aug_model(feat, feat_len, noise=noise)
        return feat

    def step(self):
        self.optimizer.step()

    def optimizer_zero_grad(self):
        self.optimizer.zero_grad()

    def load_ckpt(self, ckpt_path):
        if ckpt_path and os.path.isfile(ckpt_path):
            ckpt = torch.load(ckpt_path)
            self.aug_model.load_state_dict(ckpt['aug_model'])
            if  self.aug_type == 1:
                self.optimizer.load_state_dict(ckpt['aug_optimizer'])

    def save_ckpt(self, ckpt_path):
        if  self.aug_type == 1:
            full_dict = {
                "aug_model": self.aug_model.state_dict(),
                "aug_optimizer": self.optimizer.state_dict()
            }
        else: 
            full_dict = {"aug_model": self.aug_model.state_dict()}
        torch.save(full_dict, ckpt_path)
    def sampling_width(self):
        return self.aug_model._sampling_width()

class _TrainableAugmentModel(nn.Module):
    def __init__(self, max_T=0, T_num_masks=1, F_num_masks=1, T_position_trainable=True, F_position_trainable=True, generated_width=True, random_sample_width=False,  noise_dim=10, dim=[10, 10, 10], use_bn=False, \
    replace_with_zero=False, width_init_bias=-3., init_sigmoid_threshold=5, max_sigmoid_threshold=15, max_step = 80000, **kwargs):
        '''
        noise_dim: the input noise to the generation network
        '''
        super(_TrainableAugmentModel, self).__init__()
        self.T_num_masks = T_num_masks
        self.F_num_masks = F_num_masks
        self.max_T = max_T # if max_T is None, the width will multiplied by each sequence length 

        self.noise_dim = noise_dim
        self.dim = dim
        self.replace_with_zero = replace_with_zero
        self.width_init_bias = width_init_bias
        self.max_step = max_step
        self.max_sigmoid_threshold = max_sigmoid_threshold
        self.init_sigmoid_threshold = init_sigmoid_threshold
        self.slope = (self.max_sigmoid_threshold-self.init_sigmoid_threshold)/self.max_step
        self.step = None
        self.generated_width = generated_width

        self.output_num = (self.T_num_masks+self.F_num_masks)*2 # position and width
        assert(self.output_num>0)

        self.trainable_aug = True # whether this module trainable, or serve as a normal augmentation module

        self.all_models = []  # aug_param layout [T_mask_center, T_mask_width, F_mask_center, F_mask_width] 
        # init T_mask_center models
        for _ in range(self.T_num_masks):
            self.all_models.append(_PositionModule(position_trainable=T_position_trainable, noise_dim=noise_dim, dim=dim, use_bn=use_bn))
        # init T_mask_width models
        for _ in range(self.T_num_masks):
            self.all_models.append(_WidthModule(generated_width=generated_width, random_sample_width=random_sample_width, noise_dim=noise_dim, dim=dim, use_bn=use_bn, width_init_bias=width_init_bias))
        # init F_mask_center models
        for _ in range(self.F_num_masks):
            self.all_models.append(_PositionModule(position_trainable=F_position_trainable, noise_dim=noise_dim, dim=dim, use_bn=use_bn))
        # init F_mask_width models
        for _ in range(self.F_num_masks):
            self.all_models.append(_WidthModule(generated_width=generated_width, random_sample_width=random_sample_width, noise_dim=noise_dim, dim=dim, use_bn=use_bn, width_init_bias=width_init_bias))

        self.all_models = nn.ModuleList(self.all_models)

    def _sampling_width(self):
        if not self.generated_width:
            T_width = self.all_models[1].width.detach()
            F_width = self.all_models[3].width.detach()
            return torch.sigmoid(T_width), torch.sigmoid(F_width)
        else: 
            means = torch.zeros(1, self.noise_dim).to(self.device)
            T_width = self.all_models[1].model(means).detach()[0][0]
            F_width = self.all_models[3].model(means).detach()[0][0]
            return T_width, F_width

    def set_trainable_aug(self):
        self.trainable_aug = True
        self.train()

    def disable_trainable_aug(self):
        self.trainable_aug = False
        self.eval()

    def forward(self, feat, feat_len, noise=None):
        filling_value = 0. if self.replace_with_zero else self._get_mask_mean(feat, feat_len)
        # print('filling_value', filling_value)
        aug_param = self._generate_aug_param(feat, noise=noise)
        # print('aug_param', aug_param)

        if self.trainable_aug:
            return self._forward_trainable(feat, feat_len, filling_value, aug_param)
        else:
            return self._forward_not_trainable(feat, feat_len, filling_value, aug_param)
    
    def get_new_noise(self, feat):
        self.device = feat.device
        return torch.randn(feat.shape[0], self.noise_dim).to(feat.device)

    def _generate_aug_param(self, feat, noise=None, std=[4., 1., 4., 1.]):
        '''
        std: None or [T_center_input_std., T_center_input_std, F_center_input_std., F_center_input_std]
        '''
        if noise is None:
            noise = self.get_new_noise(feat)
        assert(len(std)==4)

        output = []
        model_idx = 0
        # forward T_mask_center models
        for _ in range(self.T_num_masks):
            output.append(self.all_models[model_idx](noise*std[0])) # [B x 1]
            model_idx += 1
        # forward T_mask_width models
        for _ in range(self.T_num_masks):
            output.append(self.all_models[model_idx](noise*std[1]))
            model_idx += 1
        # forward F_mask_center models
        for _ in range(self.F_num_masks):
            output.append(self.all_models[model_idx](noise*std[2]))
            model_idx += 1
        # forward F_mask_width models
        for _ in range(self.F_num_masks):
            output.append(self.all_models[model_idx](noise*std[3]))
            model_idx += 1
        aug_param = torch.cat(output, -1)
        return aug_param # [B, 2*(T_num_masks+F_num_masks)]

    def set_step(self, step): 
        self.step = step

    def sigmoid_threshold_scheduler(self):
        '''
        linearly ascend SIGMOID_THRESHOLD from 5 to max_sigmoid_threshold according to max_step
        
        self.max_step
        self.max_sigmoid_threshold
        '''
        assert (self.step is not None)
        curr_sigmoid_threshold = self.init_sigmoid_threshold + self.step * self.slope

        return curr_sigmoid_threshold
        
    def _forward_trainable(self, feat, feat_len, filling_value, aug_param):
        '''
        filling_value: [B]
        aug_param: [B, 2*(F_num_mask+T_num_mask)]
        '''
        def _get_soft_log_left_weight(mask_center, mask_width, length, padding_len, max_len=None):
            '''
            mask_center: [B, num_mask]
            mask_width: [B, num_mask]
            length: [B] # can be T or F dimension
            padding_len: int # length after padding
            max_len: int or None # possible length over whole data, if None, the width will multiply to length

            output: [B, l]
            '''
            SIGMOID_THRESHOLD = self.sigmoid_threshold_scheduler() # assume 2*SIGMOID_THRESHOLD is the width, because sigmoid(SIGMOID_THRESHOLD) starts to very close to 1
            device = mask_center.device

            position = torch.arange(start=0, end=padding_len, device=device).unsqueeze(0) # [1, l]

            mask_center_recover = mask_center*length.unsqueeze(-1) # [B, num_mask]
            dist_to_center = mask_center_recover.unsqueeze(-1) - position.unsqueeze(1) # [B, num_mask, l]
            mask_width_recover = mask_width*max_len if max_len else mask_width*length.unsqueeze(-1)  # [B, num_mask]
            abs_ratio = torch.abs(dist_to_center)/mask_width_recover.unsqueeze(-1) # [B, num_mask, l]

            log_mask_weight = F.logsigmoid((abs_ratio*(2*SIGMOID_THRESHOLD))-SIGMOID_THRESHOLD) # [B, num_mask, l]
            log_mask_weight = log_mask_weight.sum(1) # [B, l]
            return log_mask_weight

        # mask T
        T_log_mask_weight = _get_soft_log_left_weight(aug_param[:, :self.T_num_masks], \
                                                      aug_param[:, self.T_num_masks:2*self.T_num_masks], \
                                                      feat_len, \
                                                      feat.shape[1], self.max_T)
        # mask F
        F_log_mask_weight = _get_soft_log_left_weight(aug_param[:, 2*self.T_num_masks:2*self.T_num_masks+self.F_num_masks], \
                                                      aug_param[:, 2*self.T_num_masks+self.F_num_masks:2*self.T_num_masks+2*self.F_num_masks], \
                                                      feat_len.new_tensor([feat.shape[-1]]), \
                                                      feat.shape[-1], None)
        total_log_left_weight = T_log_mask_weight.unsqueeze(-1)+F_log_mask_weight.unsqueeze(1) # [B, T, F]
        total_left_weight = torch.exp(total_log_left_weight)
        # print(total_left_weight)

        new_feat = feat*total_left_weight + filling_value.unsqueeze(1).unsqueeze(1)*(1-total_left_weight)
        # we do not mask the fill result, cuz assume ASR model will handle this
        return new_feat

    def _forward_not_trainable(self, feat, feat_len, filling_value, aug_param):
        # use fix 
        def _get_hard_mask(mask_center, mask_width, length, padding_len, max_len=None):
            '''
            mask_center: [B, num_mask]
            mask_width: [B, num_mask]
            length: [B] # can be T or F dimension
            padding_len: int # length after padding
            max_len: int or None # possible length over whole data, if None, the width will multiply to length

            output: [B, l]
            '''
            device = mask_center.device
            position = torch.arange(start=0, end=padding_len, device=device).unsqueeze(0) # [1, l]

            mask_center_recover = mask_center*length.unsqueeze(-1) # [B, num_mask]
            mask_width_recover = mask_width*max_len if max_len else mask_width*length.unsqueeze(-1) # [B, num_mask]
            mask_left = mask_center_recover  - mask_width_recover/2. - torch.rand_like(mask_center_recover) # [B, num_mask]
            mask_right = mask_center_recover + mask_width_recover/2. + torch.rand_like(mask_center_recover) # [B, num_mask]

            mask_left  = position.unsqueeze(1) > mask_left.unsqueeze(-1) # [B, num_mask, l]
            mask_right = position.unsqueeze(1) < mask_right.unsqueeze(-1)
            mask = mask_left & mask_right
            mask = mask.any(1) # [B, l]

            return mask

        # mask T
        T_mask = _get_hard_mask(aug_param[:, :self.T_num_masks], \
                                aug_param[:, self.T_num_masks:2*self.T_num_masks], \
                                feat_len, \
                                feat.shape[1], self.max_T)
        # mask F
        F_mask = _get_hard_mask(aug_param[:, 2*self.T_num_masks:2*self.T_num_masks+self.F_num_masks], \
                                aug_param[:, 2*self.T_num_masks+self.F_num_masks:2*self.T_num_masks+2*self.F_num_masks], \
                                feat_len.new_tensor([feat.shape[-1]]), \
                                                      feat.shape[-1], None)

        total_mask = T_mask.unsqueeze(-1) | F_mask.unsqueeze(1)
        # print(total_mask)

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

class _PositionModule(nn.Module):
    def __init__(self, position_trainable=True, noise_dim=10, dim=[10, 10, 10], use_bn=False):
        super(_PositionModule, self).__init__()
        self.position_trainable = position_trainable
        if position_trainable:
            self.model = self.create_model(noise_dim, dim, use_bn)

    def create_model(self, noise_dim, dim, use_bn):
        module_list = []
        prev_dim = noise_dim
        for d in dim:
            linear = nn.Linear(prev_dim, d)
            nn.init.kaiming_normal_(linear.weight, nonlinearity='relu')
            module_list.append(linear)
            prev_dim = d
            module_list.append(nn.ReLU())
            if use_bn:
                module_list.append(nn.BatchNorm1d(d))
            
        module_list.append(nn.Linear(prev_dim, 1))
        module_list.append(nn.Sigmoid())

        return nn.Sequential(*module_list)

    def forward(self, noise):
        if self.position_trainable:
            return self.model(noise)
        else:
            return torch.rand(noise.shape[0], 1).to(noise.device)

class _WidthModule(nn.Module):
    def __init__(self, generated_width=True, random_sample_width=False, noise_dim=10, dim=[10, 10, 10], use_bn=False, width_init_bias=-3.):
        super(_WidthModule, self).__init__()
        self.generated_width = generated_width
        self.random_sample_width = random_sample_width

        if generated_width:
            self.model = self.create_model(noise_dim, dim, use_bn, width_init_bias)
        else:
            # self.width = torch.nn.parameter.Parameter(F.sigmoid(torch.tensor(width_init_bias)), requires_grad=True)
            self.width = torch.nn.parameter.Parameter(torch.tensor(width_init_bias), requires_grad=True)

    def create_model(self, noise_dim, dim, use_bn, width_init_bias):
        module_list = []
        prev_dim = noise_dim
        for d in dim:
            linear = nn.Linear(prev_dim, d)
            nn.init.kaiming_normal_(linear.weight, nonlinearity='relu')
            module_list.append(linear)
            prev_dim = d
            module_list.append(nn.ReLU())
            if use_bn:
                module_list.append(nn.BatchNorm1d(d))
            
        module_list.append(nn.Linear(prev_dim, 1))
        module_list.append(nn.Sigmoid())
        module_list[-2].bias.data.fill_(width_init_bias)

        return nn.Sequential(*module_list)

    def forward(self, noise):
        if self.generated_width:
            output = self.model(noise)
        else:
            output = self.width.unsqueeze(0).expand(noise.shape[0]).unsqueeze(-1)
            # NOT SURE
            output = torch.sigmoid(output)
        
        if self.random_sample_width:
            return output*torch.rand_like(output)
        else:
            return output

if __name__ == '__main__':
    batch_size = 3
    l = 20
    feat_dim = 5
    feat = torch.rand(batch_size, l, feat_dim).cuda()
    # feat_len = torch.randint(1, l-1, size=(batch_size,)).cuda()
    feat_len = torch.tensor([20,10,5]).cuda()

    # testing normal case
    trainable_aug = _TrainableAugmentModel(T_num_masks=1, noise_dim=100, generated_width=False, random_sample_width=True).cuda()
    trainable_aug.set_step(0)
    for i in range(10):
        trainable_aug.set_trainable_aug()
        new_feat_soft  = trainable_aug(feat, feat_len)
        trainable_aug.disable_trainable_aug()
        new_feat_hard  = trainable_aug(feat, feat_len)

    # # testing max_T
    # trainable_aug = _TrainableAugmentModel(T_num_masks=1, max_T=l, noise_dim=100).cuda()
    # trainable_aug.set_step(0)
    # new_feat_soft  = trainable_aug(feat, feat_len)
    # trainable_aug.disable_trainable_aug()
    # new_feat_hard  = trainable_aug(feat, feat_len)

    # # testing if one mask is zeros
    # trainable_aug = _TrainableAugmentModel(T_num_masks=0, noise_dim=100).cuda()
    # trainable_aug.set_step(0)
    # new_feat_soft  = trainable_aug(feat, feat_len)
    # trainable_aug.disable_trainable_aug()
    # new_feat_hard  = trainable_aug(feat, feat_len)