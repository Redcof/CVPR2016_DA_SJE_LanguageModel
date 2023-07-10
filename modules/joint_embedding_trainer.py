import gc
import os
import pathlib
import pickle
import time

import torch
from torch import optim, Tensor
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter

from modules.HybridCNN import HybridCNN
from modules.ImageEncoder import ImageEncoder
from util.model_utils import mkdir_p, weights_init, save_model
from util.text_encoder_interface import EmbeddingFactory


class JointEmbeddingTrainer:
    
    def __init__(self, output_dir, cfg, accelerate=False):
        self.model_dir = os.path.join(output_dir, 'Model')
        self.image_dir = os.path.join(output_dir, 'Image')
        self.log_dir = os.path.join(output_dir, 'Log')
        mkdir_p(self.model_dir)
        mkdir_p(self.image_dir)
        mkdir_p(self.log_dir)
        print("Output:", output_dir)
        self.summary_writer = SummaryWriter(self.log_dir)
        
        self.max_epoch = cfg.max_epoch
        if cfg.snapshot_interval == 'best':
            self.save_best = True
        else:
            self.save_best = False
            self.snapshot_interval = int(cfg.snapshot_interval)
        
        s_gpus = cfg.gpu_id.split(',')
        self.gpus = [int(ix) for ix in s_gpus]
        self.num_gpus = len(self.gpus)
        self.batch_size = cfg.batch_size * self.num_gpus
        # torch.cuda.set_device(self.gpus[0])
        cudnn.benchmark = True
        
        if accelerate:
            from accelerate import Accelerator
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
    
    def load_netTXT(self, args):
        # create models
        netTXT = HybridCNN(args.vocab_length, args.emb_dim, dropout=args.dropout)
        netTXT.apply(weights_init)
        if args.predict or args.finetune:
            state_dict = torch.load(args.NET_TXT, map_location=lambda storage, loc: storage)
            netTXT.load_state_dict(state_dict)
        netTXT.to(self.device)  # using accelerator
        return netTXT
    
    def load_netIMG(self, args):
        netIMG = ImageEncoder(args.emb_dim)
        if args.predict or args.finetune:
            state_dict = torch.load(args.NET_IMG, map_location=lambda storage, loc: storage)
            netIMG.load_state_dict(state_dict)
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
    
    def joint_embedding_loss0(self, batch_enc_txt, batch_enc_img, batch_label, class_ids):
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
        # batch x classes+1 matrix
        lv_batch = torch.zeros(batch_size, nx_classes + 1).to(self.device)
        lt_batch = torch.zeros(batch_size, nx_classes + 1).to(self.device)
        # print("Shape lv_batch:", lv_batch.shape, "lt_batch:", lt_batch.shape)
        for cls_idx, y in enumerate(class_ids):
            yny = self.delta(yn, repeat(torch.tensor(y), yn))
            # print("Shape yny:", yny.shape)
            for batch_idx, t in enumerate(batch_enc_txt):
                class_scores = yny + self.compatibility_F(vn, repeat(t, vn)).reshape(batch_size) - vntn
                lv_batch[batch_idx, :-1] = class_scores  # excluding 0
            for batch_idx, v in enumerate(batch_enc_img):
                class_scores = yny + self.compatibility_F(repeat(v, tn), tn) - vntn
                lt_batch[batch_idx, :-1] = class_scores  # excluding 0
        # batch wise all class max including 0 for both lists lv and lt, followed by average
        loss = torch.max(lv_batch, dim=1).values.sum() + torch.max(lt_batch, dim=1).values.sum() / torch.tensor(
            batch_size).to(self.device)
        # print("Shape loss:", loss.shape)
        return loss
    
    def joint_embedding_loss(self, batch_enc_txt, batch_enc_img, batch_label, class_ids):
        yn_batch = batch_label
        vn_batch = batch_enc_txt
        tn_batch = batch_enc_img
        
        def batch(low_dim: Tensor, high_dim: Tensor, axis=0):
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
        
        batch_size = yn_batch.shape[0]
        nx_classes = len(class_ids)
        vntn_batch = self.compatibility_F(vn_batch, tn_batch).reshape(batch_size)
        # print("\nShape yn_batch:", yn_batch.shape, "vn_batch:",
        # vn_batch.shape, "rn:", tn_batch.shape, "vntn_batch:", vntn_batch.shape)
        # batch x classes+1 matrix
        lv_batch = torch.zeros(batch_size, nx_classes + 1).to(self.device)
        lt_batch = torch.zeros(batch_size, nx_classes + 1).to(self.device)
        # print("Shape lv_batch:", lv_batch.shape, "lt_batch:", lt_batch.shape)
        for cls_idx, y in enumerate(class_ids):
            yny_batch = self.delta(yn_batch, batch(torch.tensor(y), yn_batch))
            # print("Shape yny_batch:", yny_batch.shape)
            for batch_idx, t in enumerate(batch_enc_txt):
                batch_scores = yny_batch + self.compatibility_F(vn_batch, batch(t, vn_batch)).reshape(
                    batch_size) - vntn_batch
                lv_batch[:, cls_idx] = batch_scores  # excluding 0
            for batch_idx, v in enumerate(batch_enc_img):
                batch_scores = yny_batch + self.compatibility_F(batch(v, tn_batch), tn_batch) - vntn_batch
                lt_batch[:, cls_idx] = batch_scores  # excluding 0
        # batch wise all class maxes including 0 for both lists lv and lt, followed by average
        loss_scaler = torch.max(lv_batch, dim=1).values.sum() + torch.max(lt_batch, dim=1).values.sum() / torch.tensor(
            batch_size).to(self.device)
        # print("Shape loss_scaler:", loss_scaler.shape)
        return loss_scaler
    
    def train(self, data_loader, args, test_dataset=None):
        with open(os.path.join(self.log_dir, "config.txt"), "w") as fp:
            args.batch_count = len(data_loader)
            with open(os.path.join(self.log_dir, "vocabulary.txt"), "w") as fp2:
                fp2.writelines(map(lambda x: "%s\n" % x, args.vocabulary))
                del args.vocabulary
            fp.write("%s" % (str(args)))
        
        # load network
        netIMG = self.load_netIMG(args)
        netTXT = self.load_netTXT(args)
        
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
        lowest_loss = float("+inf")
        save_model_now = False
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
            # #########################
            # ###### EPOCH START ######
            loop_ran = False
            for batch_idx, (txt_batch, img_batch, label_batch) in enumerate(data_loader, 0):
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
                    self.summary_writer.add_scalar('loss@step', joint_loss.data, count)
                print("\rEpoch: {}/{} Batch: {}/{} Loss: {} Memory(GB): {} ".format(
                    epoch + 1, self.max_epoch, batch_idx + 1, len(data_loader),
                    round(joint_loss.data.item(), 4), round(torch.cuda.memory_allocated() / 1e9, 4)
                ), end="\b")
            # #########################
            # ###### EPOCH END ########
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
            print('\n[%d/%d] loss: %.4f Total Time: %.2fsec'
                  % (epoch, self.max_epoch, joint_loss.data.item(), (end_t - start_t)))
            
            # check model performance
            if joint_loss.data.item() < lowest_loss:
                lowest_loss = joint_loss.data.item()
                save_model_now = True
            if (self.save_best and save_model_now) or (epoch % self.snapshot_interval == 0):
                save_model(netIMG, netTXT, epoch, self.model_dir)
                save_model_now = False
            self.summary_writer.add_scalar('loss@epoch', joint_loss.data, epoch)
            # CLEAN GPU RAM  ########################
            del joint_loss
            del txt_batch
            del img_batch
            del label_batch
            # Fix: https://discuss.pytorch.org/t/how-to-totally-free-allocate-memory-in-cuda/79590
            torch.cuda.empty_cache()
            gc.collect()
            # print("After memory_allocated(GB): ", torch.cuda.memory_allocated() / 1e9)
            # print("After memory_cached(GB): ", torch.cuda.memory_reserved() / 1e9)
            # CLEAN GPU RAM ########################
        #
        save_model(netIMG, netTXT, self.max_epoch, self.model_dir)
        #
        self.summary_writer.flush()
        self.summary_writer.close()
    
    def embed_one(self, args, embed_transform, netTXT, caption):
        batch_enc_txt = self.run_inference(args, [caption], embed_transform, netTXT)
        return batch_enc_txt.detach().cpu().numpy()[0]
    
    def run_inference(self, args, captions, embed_transform, netTXT):
        batch_txt = torch.Tensor(len(captions), args.doc_length, args.vocab_length)
        for idx, caption in enumerate(captions):
            batch_txt[idx, :, :] = embed_transform(caption)
        batch_txt.to(self.device)
        batch_enc_txt = netTXT(batch_txt)
        return batch_enc_txt.detach().cpu().numpy()
    
    def predict(self, args):
        assert os.path.isfile(args.NET_TXT), "A valid pretrained path to text " \
                                             "embedding network is required:'{}'.".format(args.NET_TXT)
        with open(os.path.join(self.log_dir, "config.txt"), "w") as fp:
            fp.write("%s" % (str(args)))
        
        # create a  initial caption encoder object
        caption_dir = pathlib.Path(args.data_dir)
        embed_transform = EmbeddingFactory(args, caption_dir).get()
        args.vocab_length = embed_transform.vocab_length
        
        # load network
        netTXT = self.load_netTXT(args)
        
        file_names = []
        embeddings = []
        
        # read captions and filenames
        for caption_file in os.listdir(caption_dir):
            with open(caption_dir / caption_file, "r") as fp:
                captions = list(map(lambda cap: cap.lower().strip(), fp.readlines()))
                filename = os.path.join("train/JPEGImages", caption_file.replace('.txt', '.jpg'))
                if args.bulk >= 1:
                    if len(captions) > args.bulk:
                        captions = captions[:args.bulk]
                    elif len(captions) < args.bulk:
                        continue
                    # caption_list.append(captions)
                    file_names.append(filename)
                    # run inference
                    batch_enc_txt = self.run_inference(args, captions, embed_transform, netTXT)
                    embeddings.append(batch_enc_txt)
                else:
                    for caption in captions:
                        file_names.append(filename)
                        # run inference
                        batch_enc_txt = self.run_inference(args, [caption], embed_transform, netTXT)
                        embeddings.append(batch_enc_txt)
        
        bulk = "bulk" if args.bulk else "no-bulk"
        
        file_name_pickle = os.path.join(self.image_dir, "filenames_{}_jemb.pickle".format(bulk))
        with open(file_name_pickle, 'wb') as fpp:
            pickle.dump(file_names, fpp)
            print("'{}' is created with {} entries".format(file_name_pickle, len(file_names)))
        
        embedding_pickle = os.path.join(self.image_dir,
                                        "embedding_{}_{}_{}_jemb.pickle".format(bulk, args.embedding_strategy,
                                                                                args.emb_dim))
        with open(embedding_pickle, 'wb') as fpp:
            pickle.dump(embeddings, fpp)
            print("'{}' is created with {} entries".format(embedding_pickle, len(file_names)))
        
        test_captions = [
            "Bag contains a gun.",
            "A gun in the center.",
            "A knife in the center.",
            "Bag contains a gun.",
            "A gun in the center.",
            "A knife in the center.",
        ]
        embeddings = []
        # run inference
        for caption in test_captions:
            batch_enc_txt = self.embed_one(args, embed_transform, netTXT, caption)
            embeddings.append(batch_enc_txt)
        embedding_pickle = os.path.join(self.image_dir,
                                        "embedding_test_{}_{}_{}_jemb.pickle".format(bulk, args.embedding_strategy,
                                                                                     args.emb_dim))
        with open(embedding_pickle, 'wb') as fpp:
            pickle.dump(embeddings, fpp)
            print("'{}' is created with {} entries".format(embedding_pickle, len(file_names)))
