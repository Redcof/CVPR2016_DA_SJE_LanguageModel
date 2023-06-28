import gc
import os
import time
from netrc import netrc

import torch
from accelerate import Accelerator
from torch import optim, autograd, nn, Tensor
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter

from python.modules.HybridCNN import HybridCNN
from python.modules.ImageEncoder import ImageEncoder
from python.util.model_utils import mkdir_p, weights_init, JointEmbeddingLoss, save_model


class JointEmbeddingTrainer:
    
    def __init__(self, output_dir, cfg, accelerate=False):
        self.model_dir = os.path.join(output_dir, 'Model')
        self.image_dir = os.path.join(output_dir, 'Image')
        self.log_dir = os.path.join(output_dir, 'Log')
        mkdir_p(self.model_dir)
        mkdir_p(self.image_dir)
        mkdir_p(self.log_dir)
        self.summary_writer = SummaryWriter(self.log_dir)
        
        self.max_epoch = cfg.max_epoch
        self.snapshot_interval = cfg.snapshot_interval
        
        s_gpus = cfg.gpu_id.split(',')
        self.gpus = [int(ix) for ix in s_gpus]
        self.num_gpus = len(self.gpus)
        self.batch_size = cfg.batch_size * self.num_gpus
        torch.cuda.set_device(self.gpus[0])
        cudnn.benchmark = True
        
        if accelerate:
            self.accelerator = Accelerator()
            self.device = self.accelerator.device
            print("Running using Accelerator.")
        else:
            self.accelerator = None
            self.device = torch.device('cuda') if cfg.cuda else torch.device('cpu')
        print("Running on {}", self.device)
        
        class GlobalParam:
            acc_batch = 0.0
            acc_smooth = 0.0
        
        self.params = GlobalParam()
    
    def load_netTXT(self, train_dataset, args):
        # create models
        netTXT = HybridCNN(train_dataset.vocab_length, args.emb_dim, dropout=args.dropout)
        netTXT.apply(weights_init)
        netTXT.to(self.device)  # using accelerator
        return netTXT
    
    def load_netIMG(self, args):
        netIMG = ImageEncoder(args.emb_dim)
        netIMG.to(self.device)  # using accelerator
        return netIMG
    
    def classifier_f(self, emb_img, emb_txt):
        """Joint embedding classifier. Equation 03 and Equation 04."""
        emb = self.compatibility_F(emb_img, emb_txt)
        max_score, max_ix = torch.max(emb.unsqueeze(0), 1)
        return max_score, max_ix
    
    def delta(self, a, b):
        """Joint embedding 0-1 loss, Δ:axb, Equation 01."""
        return torch.not_equal(a.to(self.device), b.to(self.device)).to(self.device)
    
    def compatibility_F(self, enc_img, enc_txt):
        """Input and Output embedding compatibility function. F(v, t) = θ(v)⊗φ(t),
        where θ(v), φ(t) produces image embedding and text embedding respectively."""
        # batch wise dot product
        batch_size, emb_dim = enc_img.shape
        batch_F = torch.bmm(enc_img.view(batch_size, 1, emb_dim).to(self.device),
                            enc_txt.view(batch_size, emb_dim, 1).to(self.device)).reshape(batch_size)
        # print("Shape batch_F:", batch_F.shape)
        return batch_F.to(self.device)
    
    def joint_embedding_loss(self, batch_enc_txt, batch_enc_img, batch_label, class_ids):
        yn = batch_label
        vn = batch_enc_txt
        tn = batch_enc_img
        
        def repeat(low_dim: Tensor, high_dim: Tensor, axis=0):
            """
            This function create a higher dimensional tensor form a lower dimensional tensor by repeating
            the contents in one dimension. For example, a tensor "low_dim(100 x 1536)" contains some encoding values.
            Now we have a batch of 10 encoded quantities in another tensor "high_dim(10 x 100 x 1536)". This function
            can produce a "low_dim(10 x 100 x 1536)" by repeating the original "low_dim(100 x 1536)" 10 times at the
            axis=0.
            :param low_dim:
            :param high_dim:
            :param axis:
            :return:
            """
            x = low_dim.repeat(high_dim.shape[axis]).reshape(high_dim.shape)
            return x.to(self.device)
        
        batch_size = yn.shape[0]
        nx_classes = len(class_ids)
        vntn = self.compatibility_F(vn, tn).reshape(batch_size)
        # print("\nShape yn:", yn.shape, "vn:", vn.shape, "rn:", tn.shape, "vntn:", vntn.shape)
        lv_list = torch.zeros(batch_size, nx_classes + 1).to(self.device)
        lt_list = torch.zeros(batch_size, nx_classes + 1).to(self.device)
        # print("Shape lv_list:", lv_list.shape, "lt_list:", lt_list.shape)
        for cls_idx, y in enumerate(class_ids):
            yny = self.delta(yn, repeat(torch.tensor(y), yn))
            # print("Shape yny:", yny.shape)
            for batch_idx, t in enumerate(batch_enc_txt):
                batch_val = yny + self.compatibility_F(vn, repeat(t, vn)).reshape(batch_size) - vntn
                lv_list[batch_idx, :-1] = batch_val
            for batch_idx, v in enumerate(batch_enc_img):
                batch_val = yny + self.compatibility_F(repeat(v, tn), tn) - vntn
                lt_list[batch_idx, :-1] = batch_val
        loss = torch.max(lv_list, dim=1).values.sum() + torch.max(lt_list, dim=1).values.sum() / torch.tensor(
            batch_size).to(self.device)
        # print("Shape loss:", loss.shape)
        return loss
    
    def train(self, data_loader, args, test_dataset=None):
        # load network
        netIMG = self.load_netIMG(args)
        netTXT = self.load_netTXT(data_loader.dataset, args)
        
        generator_lr = args.lr_img
        discriminator_lr = args.lr_txt
        lr_decay_step = args.lr_decay
        optimizerIMG = optim.RMSprop(netIMG.get_learnable_params(), lr=generator_lr)
        optimizerTXT = optim.RMSprop(netTXT.get_learnable_params(), lr=discriminator_lr)
        
        # setup epoch
        epoch_start = 0
        if args.finetune:
            epoch_start = args.epoch_start
        ####################################
        # Update Double Point precision
        ####################################
        if self.accelerator is None:
            dtype = data_loader.dataset.dtype
            if torch.float64 == dtype:
                netIMG.double()
                netTXT.double()
        count = 0
        print("Training...")
        if self.accelerator:
            # accelerator code patch
            netIMG, optimizerIMG, netTXT, optimizerTXT, data_loader = self.accelerator.prepare(netIMG, optimizerIMG,
                                                                                               netTXT, optimizerTXT,
                                                                                               data_loader)
        for epoch in range(epoch_start, self.max_epoch):
            start_t = time.time()
            # # LR decay for Optimizers
            # if epoch % lr_decay_step == 0 and epoch > 0:
            #     generator_lr *= 0.5
            #     for param_group in optimizerIMG.param_groups:
            #         param_group['lr'] = generator_lr
            #     discriminator_lr *= 0.5
            #     for param_group in optimizerTXT.param_groups:
            #         param_group['lr'] = discriminator_lr
            
            loop_ran = False
            for batch_idx, (txt_batch, img_batch, label_batch) in enumerate(data_loader, 0):
                print("memory_allocated(GB): ", torch.cuda.memory_allocated() / 1e9)
                print("memory_cached(GB): ", torch.cuda.memory_reserved() / 1e9)
                print("\rEpoch: {}/{} Batch: {}/{} ".format(epoch + 1, self.max_epoch, batch_idx + 1, len(data_loader)),
                      end="\b")
                loop_ran = True
                #############################
                # (1) Prepare training data
                #############################
                txt_batch = txt_batch.to(self.device)  # using accelerator
                img_batch = img_batch.to(self.device)  # using accelerator
                label_batch = label_batch.to(self.device)  # using accelerator
                
                #############################
                # (2) Forward Pass
                #############################
                optimizerIMG.zero_grad()
                enc_img = netIMG(img_batch)
                
                optimizerTXT.zero_grad()
                enc_txt = netTXT(txt_batch)
                
                ###########################
                # (3) Calculate Loss
                ###########################
                joint_loss = self.joint_embedding_loss(enc_img, enc_txt, label_batch, data_loader.dataset.class_ids)
                
                ###########################
                # (4) Update network
                ###########################
                # calculate gradients
                if self.accelerator:
                    self.accelerator.backward(joint_loss)
                else:
                    joint_loss.backward()
                # update weights
                optimizerIMG.step()
                optimizerTXT.step()
                
                count = count + 1
                if batch_idx % 100 == 0:
                    self.summary_writer.add_scalar('loss', joint_loss.data, count)
                if (epoch % self.snapshot_interval == 0 or epoch == self.max_epoch - 1) and batch_idx % 100 == 0:
                    # save the image result for each epoch
                    # save_img_results(None, lr_fake, epoch, self.image_dir)
                    ###########################
                    # GENERATE TEST IMAGES
                    ###########################
                    # self.test(netIMG, test_dataset.embeddings, self.image_dir, epoch)
                    ...
            
            if loop_ran is False:
                raise Warning(
                    "Not enough data available.\n"
                    "Reasons:\n"
                    "(1) Dataset() length=0 or \n"
                    "(2) When `drop_last=True` in Dataloader() and the `Dataset() length` < `batch-size`\n"
                    "Solutions:\n"
                    "(1) Reduce batch size to satisfy `Dataset() length` >= `batch-size`[recommended]\n"
                    "(2) Set `drop_last=False`[not recommended]")
            end_t = time.time()
            print('\n[%d/%d] Loss_D: %.4f Total Time: %.2fsec'
                  % (epoch, self.max_epoch, joint_loss.data.item(), (end_t - start_t)))
            
            if epoch % self.snapshot_interval == 0:
                save_model(netIMG, netTXT, epoch, self.model_dir)
            
            # CLEAN GPU RAM  ########################
            del joint_loss
            del txt_batch
            del img_batch
            del label_batch
            # Fix: https://discuss.pytorch.org/t/how-to-totally-free-allocate-memory-in-cuda/79590
            torch.cuda.empty_cache()
            gc.collect()
            print("After memory_allocated(GB): ", torch.cuda.memory_allocated() / 1e9)
            print("After memory_cached(GB): ", torch.cuda.memory_reserved() / 1e9)
            # CLEAN GPU RAM ########################
        #
        save_model(netIMG, netTXT, self.max_epoch, self.model_dir)
        #
        self.summary_writer.flush()
        self.summary_writer.close()
