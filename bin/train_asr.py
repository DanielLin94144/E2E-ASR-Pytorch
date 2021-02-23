import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import yaml

from src.solver import BaseSolver

from src.asr import ASR
from src.optim import Optimizer
from src.data import load_dataset, load_wav_dataset, load_babel_dataset
from src.util import human_format, cal_er, feat_to_fig, LabelSmoothingLoss
from src.audio import Delta, Postprocess, Augment
from src.augmentation import TrainableAugment
from src.search import Search, FasterSearch
from src.collect_batch import HALF_BATCHSIZE_AUDIO_LEN

EMPTY_CACHE_STEP = 100

class Solver(BaseSolver):
    ''' Solver for training'''
    def __init__(self,config,paras,mode):
        super().__init__(config,paras,mode)

        # Curriculum learning affects data loader
        self.curriculum = self.config['hparas']['curriculum']
        self.val_mode = self.config['hparas']['val_mode'].lower()
        self.WER = 'per' if self.val_mode == 'per' else 'wer'
        

        # put specaug config
        if ('augmentation' in self.config):
            if self.config['augmentation']['type'] == 3: # specaug
                self.config['data']['audio']['augment'] = self.config['augmentation']['specaug']
            else:
                self.config['data']['audio']['augment'] = False
            # only train aug for type 1
            if self.config['augmentation']['type'] == 1:
                self.train_aug = True
            else:
                self.train_aug = False
        else:
            self.train_aug = False
            self.config['data']['audio']['augment'] = False
            self.config['augmentation'] = {'type':4, 'trainable_aug':{'model':None, 'optimizer':None}}

    def fetch_data(self, data, train=False):
        ''' Move data to device and compute text seq. length'''
        # feat: B x T x D
        
        if self.paras.babel is not None: 
            feat, feat_len, txt, txt_len = data
            # self.specaug.to(device)

            def specaug_babel(feat):
                if train and self.specaug:
                    feat = [self.specaug(f) for f in feat]
                    feat = pad_sequence(feat, batch_first=True)
                return feat
            
            txt = pad_sequence(txt, batch_first=True)
            feat = feat.to(self.device)
            feat_len = feat_len.to(self.device)
            txt = txt.to(self.device)
            
            feat = specaug_babel(feat)

            return feat, feat_len, txt, txt_len
        
        elif self.paras.upstream is not None:
            # feat is raw waveform
            device = 'cpu' if self.paras.deterministic else self.device
            self.upstream.to(device)

            def to_device(feat):
                return [f.to(device) for f in feat]

            def extract_feature(feat):
                feat = self.upstream(to_device(feat))
                if train and self.specaug and 'aug' not in self.paras.upstream:
                    feat = [self.specaug(f) for f in feat]
                return feat

            if HALF_BATCHSIZE_AUDIO_LEN < 3500 and train:
                first_len = extract_feature(feat[:1])[0].shape[0]
                if first_len > HALF_BATCHSIZE_AUDIO_LEN:
                    feat = feat[::2]
                    txt = txt[::2]

            if self.paras.upstream_trainable:
                self.upstream.train()
                feat = extract_feature(feat)
            else:
                with torch.no_grad():
                    self.upstream.eval()
                    feat = extract_feature(feat)

            feat_len = torch.LongTensor([len(f) for f in feat])
            feat = pad_sequence(feat, batch_first=True)
            txt = pad_sequence(txt, batch_first=True)
            
        else: 
            _, feat, feat_len, txt = data

        feat = feat.to(self.device)
        feat_len = feat_len.to(self.device)
        txt = txt.to(self.device)
        txt_len = torch.sum(txt!=0,dim=-1)

        return feat, feat_len, txt, txt_len

    def load_data(self):
        ''' Load data for training/validation, store tokenizer and input/output shape'''
        if self.paras.upstream is not None:
            print(f'[Solver] - using S3PRL {self.paras.upstream}')
            self.tr_set, self.dv_set, self.vocab_size, self.tokenizer, msg = \
                            load_wav_dataset(self.paras.njobs, self.paras.gpu, self.paras.pin_memory, 
                                        self.curriculum>0,
                                        **self.config['data'])
            self.upstream = torch.hub.load(
                's3prl/s3prl',
                self.paras.upstream,
                feature_selection = self.paras.upstream_feature_selection,
                refresh = self.paras.upstream_refresh,
                ckpt = self.paras.upstream_ckpt,
                force_reload = True,
            )
            self.feat_dim = self.upstream.get_output_dim()
            augment = self.config['data']['audio'].pop("augment", False)
            if augment:
                self.specaug = Augment(**augment)
            else:
                self.specaug = None

        elif self.paras.babel is not None:
            print(f'[Solver] - using babel dataset')
            self.tr_set, self.dv_set, self.vocab_size, self.tokenizer, msg = \
                            load_babel_dataset(self.paras.njobs, self.paras.gpu, self.paras.pin_memory, 
                                        self.curriculum>0, self.config['augmentation']['trainable_aug']['model']['max_T'],
                                        **self.config['data'])
            self.feat_dim = self.config['data']['audio']['feat_dim']
            self.verbose(msg)
            augment = self.config['data']['audio'].pop("augment", False)
            if augment:
                self.specaug = Augment(**augment)
            else:
                self.specaug = None

        else:
            self.tr_set, self.dv_set, self.feat_dim, self.vocab_size, self.tokenizer, msg = \
                         load_dataset(self.paras.njobs, self.paras.gpu, self.paras.pin_memory, 
                                      self.curriculum>0,
                                      **self.config['data'])
        self.verbose(msg)

        # Dev set sames
        self.dv_names = []
        if type(self.dv_set) is list:
            for ds in self.config['data']['corpus']['dev_split']:
                self.dv_names.append(ds[0])
        else:
            self.dv_names = self.config['data']['corpus']['dev_split'][0]
        
        # Logger settings
        if type(self.dv_names) is str:
            self.best_wer = {'att':{self.dv_names:3.0},
                             'ctc':{self.dv_names:3.0}}
        else:
            self.best_wer = {'att': {},'ctc': {}}
            for name in self.dv_names:
                self.best_wer['att'][name] = 3.0
                self.best_wer['ctc'][name] = 3.0

    def set_model(self):
        ''' Setup ASR model and optimizer '''

        batch_size = self.config['data']['corpus']['batch_size']//2
        
        self.model = ASR(self.feat_dim, self.vocab_size, batch_size, **self.config['model']).to(self.device)
        self.aug_model = TrainableAugment(self.config['augmentation']['type'], \
                                        self.config['augmentation']['trainable_aug']['model'], \
                                        self.config['augmentation']['trainable_aug']['optimizer']).to(self.device)
        
        
        aug_type = self.config['augmentation']['type']
        print(f'[Augmentation INFO] - augmentation type : {aug_type}')
    
            # create search object
        if self.train_aug:
            use_faster_search = self.config['augmentation'].pop('faster_search', False)

            if use_faster_search:
                print('[Augmentation INFO] - use faster search : Yes')
                self.search = FasterSearch(self.model, self.aug_model, self.config['hparas'], **self.config['augmentation']['trainable_aug']['fast_search'])
            else:
                print('[Augmentation INFO] - use faster search : No')
                self.search =       Search(self.model, self.aug_model, self.config['hparas'], **self.config['augmentation']['trainable_aug']['search'])

        self.verbose(self.model.create_msg())
        model_paras = [{'params':self.model.parameters()}]

        # Losses
        
        '''label smoothing'''
        if self.config['hparas']['label_smoothing']:
            self.seq_loss = LabelSmoothingLoss(31, 0.1)   
            print('[INFO]  using label smoothing. ') 
        else:    
            self.seq_loss = torch.nn.CrossEntropyLoss(ignore_index=0)
        ### zero_infinity=True
        self.ctc_loss = torch.nn.CTCLoss(blank=0, zero_infinity=True) # Note: zero_infinity=False is unstable?

        # Optimizer
        self.optimizer = Optimizer(model_paras, **self.config['hparas'])
        self.lr_scheduler = self.optimizer.lr_scheduler
        self.verbose(self.optimizer.create_msg())

        # Enable AMP if needed
        self.enable_apex()
    
        """
        support resume training 
        self.paras.load: model PATH
        self.paras.load_aug: aug model PATH
        """
        if self.paras.load: 
            self.load_ckpt() # model, model optimizer, step, performance
    
        if self.paras.load_aug:
            self.aug_model.load_ckpt(self.paras.load_aug) # aug model 


    def calc_asr_loss(self, ctc_output, encode_len, att_output, txt, txt_len, stop_step):
        total_loss = 0
        ctc_loss = None
        att_loss = None
        if self.early_stoping:
            if self.step > stop_step:
                ctc_output = None
                self.model.ctc_weight = 0
        
        # Compute all objectives
        if ctc_output is not None:
            if self.paras.cudnn_ctc:
                ctc_loss = self.ctc_loss(ctc_output.transpose(0,1), 
                                            txt.to_sparse().values().to(device='cpu',dtype=torch.int32),
                                            [ctc_output.shape[1]]*len(ctc_output),
                                            #[int(encode_len.max()) for _ in encode_len],
                                            txt_len.cpu().tolist())
            else:
                ctc_loss = self.ctc_loss(ctc_output.transpose(0,1), txt, encode_len, txt_len)
            total_loss += ctc_loss*self.model.ctc_weight
            del encode_len

        if att_output is not None:
            #print(att_output.shape)
            b,t,_ = att_output.shape
            att_loss = self.seq_loss(att_output.view(b*t,-1),txt.view(-1))
            # Sum each uttr and devide by length then mean over batch
            # att_loss = torch.mean(torch.sum(att_loss.view(b,t),dim=-1)/torch.sum(txt!=0,dim=-1).float())
            total_loss += att_loss*(1-self.model.ctc_weight)
        
        return total_loss, ctc_loss, att_loss

    def exec(self):
        ''' Training End-to-end ASR system '''
        self.verbose('Total training steps {}.'.format(human_format(self.max_step)))
        
        self.n_epochs = 0
        self.timer.set()
        '''early stopping for ctc '''
        self.early_stoping = self.config['hparas']['early_stopping']
        stop_epoch = 10
        batch_size = self.config['data']['corpus']['batch_size']
        stop_step = len(self.tr_set)*stop_epoch//batch_size
        


        while self.step< self.max_step:
            ctc_loss, att_loss, emb_loss = None, None, None
            # Renew dataloader to enable random sampling 
            
            if self.curriculum>0 and n_epochs==self.curriculum:
                self.verbose('Curriculum learning ends after {} epochs, starting random sampling.'.format(n_epochs))
                self.tr_set, _, _, _, _, _ = \
                         load_dataset(self.paras.njobs, self.paras.gpu, self.paras.pin_memory, 
                                      False, **self.config['data'])
            
            
            for tr_data in self.tr_set:
                # Pre-step : update tf_rate/lr_rate and do zero_grad
                tf_rate = self.optimizer.pre_step(self.step)
            
                # Fetch data
                feat, feat_len, txt, txt_len = self.fetch_data(tr_data, train=True)                
                self.timer.cnt('rd')
                # Forward model
                # Note: txt should NOT start w/ <sos>
                noise = self.aug_model.get_new_noise(feat)                
                aug_feat = self.aug_model(feat, feat_len, noise=noise) 
                ctc_output, encode_len, att_output, att_align, dec_state = \
                    self.model( aug_feat, feat_len, max(txt_len), tf_rate=tf_rate,
                                    teacher=txt, get_dec_state=False)

                # Clear not used objects
                del att_align
                del dec_state

                total_loss, ctc_loss, att_loss = self.calc_asr_loss(ctc_output, encode_len, att_output, txt, txt_len, stop_step)

                self.timer.cnt('fw')

                # Backprop
                grad_norm = self.backward(total_loss)             

                
                
                # train aug
                # TODO add time counter for train aug
                if self.train_aug:
                    # NOT SURE: do not need to zero grad aug_model in the first
                    dv_data = next(iter(self.dv_set))
                    train_data = [feat, feat_len, txt, txt_len, tf_rate, stop_step]
                    valid_data = [*self.fetch_data(dv_data), tf_rate, stop_step]
                    self.search.unrolled_backward(train_data, valid_data, self.optimizer.opt.param_groups[0]['lr'], self.optimizer.opt, self.calc_asr_loss, noise=noise)
                    noise = None
                    self.aug_model.step()
                    self.aug_model.optimizer_zero_grad()

                    self.aug_model.aug_model.set_step(self.step)
                    
                    if self.step % self.valid_step == 0: 
                        print(f'[INFO] - sigmoid threshold = {self.aug_model.aug_model.SIGMOID_THRESHOLD} @ step {self.step}')

                self.timer.cnt('aug')

                self.step+=1

                # Logger
                if (self.step==1) or (self.step%self.PROGRESS_STEP==0):
                    self.progress('Tr stat | Loss - {:.2f} | Grad. Norm - {:.2f} | {}'\
                            .format(total_loss.cpu().item(),grad_norm,self.timer.show()))
                    self.write_log('emb_loss',{'tr':emb_loss})
                    if att_output is not None:
                        self.write_log('loss',{'tr_att':att_loss})
                        self.write_log(self.WER,{'tr_att':cal_er(self.tokenizer,att_output,txt)})
                        self.write_log(   'cer',{'tr_att':cal_er(self.tokenizer,att_output,txt,mode='cer')})
                    if ctc_output is not None:
                        self.write_log('loss',{'tr_ctc':ctc_loss})
                        self.write_log(self.WER,{'tr_ctc':cal_er(self.tokenizer,ctc_output,txt,ctc=True)})
                        self.write_log(   'cer',{'tr_ctc':cal_er(self.tokenizer,ctc_output,txt,mode='cer',ctc=True)})
                        self.write_log('ctc_text_train',self.tokenizer.decode(ctc_output[0].argmax(dim=-1).tolist(),
                                                                                                ignore_repeat=True))
                    # if self.step==1 or self.step % (self.PROGRESS_STEP * 5) == 0:
                    #     self.write_log('spec_train',feat_to_fig(feat[0].transpose(0,1).cpu().detach(), spec=True))
                    #del total_loss

                # Validation
                if (self.step==1) or (self.step%self.valid_step == 0):
                    if type(self.dv_set) is list:
                        for dv_id in range(len(self.dv_set)):
                            self.validate(self.dv_set[dv_id], self.dv_names[dv_id])
                    else:
                        self.validate(self.dv_set, self.dv_names)
                if self.step % (len(self.tr_set)// batch_size)==0: # one epoch
                    # print('Have finished epoch: ', self.n_epochs)
                    self.n_epochs +=1
                    
                if self.lr_scheduler == None:
                    lr = self.optimizer.opt.param_groups[0]['lr']
                    
                    if self.step == 1:
                        print('[INFO]    using lr schedular defined by Daniel, init lr = ', lr)

                    if self.step >99999 and self.step%2000==0:
                        lr = lr*0.85
                        for param_group in self.optimizer.opt.param_groups:
                            param_group['lr'] = lr
                        print('[INFO]     at step:', self.step )
                        print('[INFO]   lr reduce to', lr)


                    #self.lr_scheduler.step(total_loss)
                # End of step
                # if self.step % EMPTY_CACHE_STEP == 0:
                    # Empty cuda cache after every fixed amount of steps
                torch.cuda.empty_cache() # https://github.com/pytorch/pytorch/issues/13246#issuecomment-529185354
                self.timer.set()
                if self.step > self.max_step: break
            
            
            
            #update lr_scheduler
            
            
        self.log.close()
        print('[INFO] Finished training after', human_format(self.max_step), 'steps.')
        
    def validate(self, _dv_set, _name):
        # Eval mode
        self.model.eval()
        # if self.emb_decoder is not None: self.emb_decoder.eval()
        dev_wer = {'att':[],'ctc':[]}
        dev_cer = {'att':[],'ctc':[]}
        dev_er  = {'att':[],'ctc':[]}

        for i,data in enumerate(_dv_set):
            self.progress('Valid step - {}/{}'.format(i+1,len(_dv_set)))
            # Fetch data
            feat, feat_len, txt, txt_len = self.fetch_data(data)

            # Forward model
            with torch.no_grad():
                ctc_output, encode_len, att_output, att_align, dec_state = \
                    self.model( feat, feat_len, int(max(txt_len)*self.DEV_STEP_RATIO), 
                                    emb_decoder=self.emb_decoder)

            if att_output is not None:
                dev_wer['att'].append(cal_er(self.tokenizer,att_output,txt,mode='wer'))
                dev_cer['att'].append(cal_er(self.tokenizer,att_output,txt,mode='cer'))
                dev_er['att'].append(cal_er(self.tokenizer,att_output,txt,mode=self.val_mode))
            if ctc_output is not None:
                dev_wer['ctc'].append(cal_er(self.tokenizer,ctc_output,txt,mode='wer',ctc=True))
                dev_cer['ctc'].append(cal_er(self.tokenizer,ctc_output,txt,mode='cer',ctc=True))
                dev_er['ctc'].append(cal_er(self.tokenizer,ctc_output,txt,mode=self.val_mode,ctc=True))
            
            # Show some example on tensorboard
            if i == len(_dv_set)//2:
                for i in range(min(len(txt),self.DEV_N_EXAMPLE)):
                    if self.step==1:
                        self.write_log('true_text_{}_{}'.format(_name, i),self.tokenizer.decode(txt[i].tolist()))
                    if att_output is not None:
                        self.write_log('att_align_{}_{}'.format(_name, i),feat_to_fig(att_align[i,0,:,:].cpu().detach()))
                        self.write_log('att_text_{}_{}'.format(_name, i),self.tokenizer.decode(att_output[i].argmax(dim=-1).tolist()))
                    if ctc_output is not None:
                        self.write_log('ctc_text_{}_{}'.format(_name, i),self.tokenizer.decode(ctc_output[i].argmax(dim=-1).tolist(),
                                                                                                       ignore_repeat=True))
        
        # Ckpt if performance improves
        tasks = []
        if len(dev_er['att']) > 0:
            tasks.append('att')
        if len(dev_er['ctc']) > 0:
            tasks.append('ctc')

        for task in tasks:
            dev_er[task] = sum(dev_er[task])/len(dev_er[task])
            dev_wer[task] = sum(dev_wer[task])/len(dev_wer[task])
            dev_cer[task] = sum(dev_cer[task])/len(dev_cer[task])
            if dev_er[task] < self.best_wer[task][_name]:
                self.best_wer[task][_name] = dev_er[task]
                self.save_checkpoint('best_{}_{}.pth'.format(task, _name + (self.save_name if self.transfer_learning else '')), 
                                    self.val_mode,dev_er[task],_name)
                # save aug model ckpt 
                if self.config['augmentation']['type'] != 4 and self.config['augmentation']['type'] != 3:
                    self.aug_model.save_ckpt(self.ckpdir+'/best_aug.pth')

            if self.step >= self.max_step:
                self.save_checkpoint('last_{}_{}.pth'.format(task, _name + (self.save_name if self.transfer_learning else '')), 
                                    self.val_mode,dev_er[task],_name)
                if self.config['augmentation']['type'] != 4 and self.config['augmentation']['type'] != 3:
                    self.aug_model.save_ckpt(self.ckpdir+'/last_aug.pth')

            self.write_log(self.WER,{'dv_'+task+'_'+_name.lower():dev_wer[task]})
            self.write_log(   'cer',{'dv_'+task+'_'+_name.lower():dev_cer[task]})
            
        # Resume training
        self.model.train()

