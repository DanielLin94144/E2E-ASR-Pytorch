import copy
import torch

class Search():
    """ Compute gradients of alphas """
    def __init__(self, model, aug_model, model_optim_parameter, forward_optim_method='sgd'):
        """
        Args:
            model
            w_momentum: weights momentum
        """
        print('creating search object')
        self.model = model
        self.v_model = copy.deepcopy(model) # contain the parameter of the virtual step
        self.model_optim_parameter = model_optim_parameter
        # replace by model_optim_parameter
        #self.w_momentum = w_momentum
        #self.w_weight_decay = w_weight_decay

        self.aug_model = aug_model
        self.forward_optim_method = forward_optim_method # the method to do virtual step

    def virtual_step(self, train_data, model_lr, model_optim, get_loss_func):
        """
        Compute unrolled weight w' (virtual step)

        Step process:
        1) forward
        2) calc loss
        3) compute gradient (by backprop)
        4) update gradient

        Args:
            model_lr: learning rate for virtual gradient step (same as weights lr)
            model_optim: weights optimizer
        """
        aug_train_data = self.aug_train_data(train_data)

        # forward & calc loss
        model_output = self.model(*self._get_model_input(aug_train_data))
        loss = get_loss_func(*self._get_calc_loss_input(train_data, model_output)) # L_trn(w)

        # compute gradient
        gradients = torch.autograd.grad(loss, self.model.parameters())

        # do virtual step (update gradient)
        # below operations do not need gradient tracking
        with torch.no_grad():
            # dict key is not the value, but the pointer. So original modelwork weight have to
            # be iterated also.
            for w, vw, g in zip(self.model.parameters(), self.v_model.parameters(), gradients):
                # m = model_optim.state[w].get('momentum_buffer', 0.) * self.w_momentum
                # vw.copy_(w - model_lr * (m + g + self.w_weight_decay*w))
                # TODO: consider other optimization process # here use SGD
                vw.copy_(w - model_lr * eval(self.forward_optim_method+"_forward")(w, g, model_optim, self.model_optim_parameter))
                # TODO add backward method here

    def aug_train_data(self, train_data):
        """ do augmentation on train data
        Args:
            train_data: [feat, feat_len, .....]
        """
        return [self.aug_model(train_data[0], train_data[1], noise=self.noise), *train_data[1:]]

    def unrolled_backward(self, train_data, valid_data, model_lr, model_optim, get_loss_func, noise='dummy'):
        """ Compute unrolled loss and backward its gradients
        Args:
            train_data, valid_data: data b4 aug, [feat, feat_len, .....]
            train_data should be b4 augmentation
            model_lr: learning rate for virtual gradient step (same as model lr)
            model_optim: weights optimizer - for virtual step
            get_loss_func: the function that take the output of model, and get the loss
        """
        self.noise = self.aug_model.get_new_noise(train_data[0]) # make the noise be consistent over the whole process

        # do virtual step (calc w`)
        self.virtual_step(train_data, model_lr, model_optim, get_loss_func)

        # calc unrolled loss
        model_output = self.v_model(*self._get_model_input(valid_data))
        loss = get_loss_func(*self._get_calc_loss_input(valid_data, model_output)) # L_trn(w)

        # compute gradient
        dw = torch.autograd.grad(loss, self.v_model.parameters())

        hessian = self.compute_hessian(dw, train_data, get_loss_func)

        # update final gradient = dalpha - model_lr*hessian
        with torch.no_grad():
            for alpha, h in zip(self.aug_model.parameters(), hessian):
                alpha.grad = -model_lr*h
        
        self.noise = None

    def compute_hessian(self, dw, train_data, get_loss_func):
        """
        dw = dw` { L_val(w`, alpha) }
        w+ = w + eps * dw
        w- = w - eps * dw
        hessian = (dalpha { L_trn(w+, alpha) } - dalpha { L_trn(w-, alpha) }) / (2*eps)
        eps = 0.01 / ||dw||
        """
        norm = torch.cat([w.view(-1) for w in dw]).norm()
        eps = 0.01 / norm

        # w+ = w + eps*dw`
        with torch.no_grad():
            for p, d in zip(self.model.parameters(), dw):
                p += eps * d
        aug_train_data = self.aug_train_data(train_data)
        model_output = self.model(*self._get_model_input(aug_train_data))
        loss = get_loss_func(*self._get_calc_loss_input(train_data, model_output)) # L_trn(w)
        dalpha_pos = torch.autograd.grad(loss, self.aug_model.parameters()) # dalpha { L_trn(w+) }

        # w- = w - eps*dw`
        with torch.no_grad():
            for p, d in zip(self.model.parameters(), dw):
                p -= 2. * eps * d
        aug_train_data = self.aug_train_data(train_data)
        model_output = self.model(*self._get_model_input(aug_train_data))
        loss = get_loss_func(*self._get_calc_loss_input(train_data, model_output)) # L_trn(w)
        dalpha_neg = torch.autograd.grad(loss, self.aug_model.parameters()) # dalpha { L_trn(w-) }

        # recover w
        with torch.no_grad():
            for p, d in zip(self.model.parameters(), dw):
                p += eps * d

        hessian = [(p-n) / (2.*eps) for p, n in zip(dalpha_pos, dalpha_neg)]
        return hessian

    def _get_model_input(self, data):
        feat, feat_len, txt, txt_len, tf_rate, stop_step = data
        return [feat, feat_len, max(txt_len), tf_rate, txt] 

    def _get_calc_loss_input(self, data, model_output):
        feat, feat_len, txt, txt_len, tf_rate, stop_step = data
        ctc_output, encode_len, att_output, att_align, dec_state = model_output
        return [ctc_output, encode_len, att_output, txt, txt_len, stop_step]

class FasterSearch():
    """ Compute gradients of alphas """
    def __init__(self, model, aug_model, model_optim_parameter, update_valid_frequency=1, update_valid_weight=1.):
        """
        Args:
            model
            w_momentum: weights momentum
        """
        print('creating faster search object')
        self.model = model
        self.v_model = copy.deepcopy(model) # contain the parameter of the previous model
        self.model_optim_parameter = model_optim_parameter
        # replace by model_optim_parameter
        #self.w_momentum = w_momentum
        #self.w_weight_decay = w_weight_decay

        self.aug_model = aug_model

        self.update_valid_frequency = update_valid_frequency
        self.update_valid_weight = update_valid_weight

        self.valid_counter = update_valid_frequency
        self.valid_gradient = None

    def copy_net(self):
        with torch.no_grad():
            # dict key is not the value, but the pointer. So original modelwork weight have to
            # be iterated also.
            for w, vw in zip(self.model.parameters(), self.v_model.parameters()):
                vw.copy_(w)

    def aug_train_data(self, train_data):
        """ do augmentation on train data
        Args:
            train_data: [feat, feat_len, .....]
        """
        return [self.aug_model(train_data[0], train_data[1], noise=self.noise), *train_data[1:]]

    def unrolled_backward(self, train_data, valid_data, model_lr, model_optim, get_loss_func, noise=None):
        """ Compute unrolled loss and backward its gradients
        Args:
            train_data, valid_data: data b4 aug, [feat, feat_len, .....]
            train_data should be b4 augmentation
            model_lr: learning rate for virtual gradient step (same as model lr)
            model_optim: weights optimizer - for virtual step
            get_loss_func: the function that take the output of model, and get the loss
        """
        self.noise = noise # noise should be the same with the previous noise to train model

        # calc unrolled loss
        if self.valid_counter%self.update_valid_frequency == 0: # update self.valid_gradient
            model_output = self.model(*self._get_model_input(valid_data))
            loss = get_loss_func(*self._get_calc_loss_input(valid_data, model_output)) # L_trn(w)
            # compute gradient
            dw = torch.autograd.grad(loss, self.model.parameters())
            # update weighted sum
            if self.valid_gradient:
                assert len(dw) == len(self.valid_gradient)
                self.valid_gradient = tuple([self.update_valid_weight*current+(1-self.update_valid_weight)*previous for current, previous in zip(dw, self.valid_gradient)])
            else:
                self.valid_gradient = dw
            self.valid_counter = 1
        else:
            self.valid_counter += 1

        hessian = self.compute_hessian(self.valid_gradient, train_data, get_loss_func)

        # update final gradient = dalpha - model_lr*hessian
        with torch.no_grad():
            for alpha, h in zip(self.aug_model.parameters(), hessian):
                alpha.grad = -model_lr*h
        
        self.noise = None
        self.copy_net()

    def compute_hessian(self, dw, train_data, get_loss_func):
        """
        dw = dw` { L_val(w`, alpha) }
        w+ = w + eps * dw
        w- = w - eps * dw
        hessian = (dalpha { L_trn(w+, alpha) } - dalpha { L_trn(w-, alpha) }) / (2*eps)
        eps = 0.01 / ||dw||
        """
        norm = torch.cat([w.view(-1) for w in dw]).norm()
        eps = 0.01 / norm

        dalpha_neg = tuple([x.grad for x in self.aug_model.parameters()]) # dalpha { L_trn(w-) }

        # w+ = w + eps*dw`
        with torch.no_grad():
            for p, d in zip(self.v_model.parameters(), dw):
                p += eps * d
        aug_train_data = self.aug_train_data(train_data)
        model_output = self.v_model(*self._get_model_input(aug_train_data))
        loss = get_loss_func(*self._get_calc_loss_input(train_data, model_output)) # L_trn(w)
        dalpha_pos = torch.autograd.grad(loss, self.aug_model.parameters()) # dalpha { L_trn(w+) }

        hessian = [(p-n) / eps for p, n in zip(dalpha_pos, dalpha_neg)]
        return hessian

    def _get_model_input(self, data):
        feat, feat_len, txt, txt_len, tf_rate, stop_step = data
        return [feat, feat_len, max(txt_len), tf_rate, txt] 

    def _get_calc_loss_input(self, data, model_output):
        feat, feat_len, txt, txt_len, tf_rate, stop_step = data
        ctc_output, encode_len, att_output, att_align, dec_state = model_output
        return [ctc_output, encode_len, att_output, txt, txt_len, stop_step]

def sgd_forward(w, g, model_optim, model_optim_parameter):
    return g

# def sgd_backward(w, g, model_optim, model_optim_parameter):
#     return 1

def momentum_forward(w, g, model_optim, model_optim_parameter):
    m = model_optim.state[w].get('momentum_buffer', 0.) * model_optim_parameter['w_momentum']
    return m + g + model_optim_parameter['w_weight_decay']*w
    
# def momentum_backward(w, g, model_optim, model_optim_parameter):
#     return 1