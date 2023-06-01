import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import os
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10


parser = argparse.ArgumentParser(description='Train a multi-modal embedding model')

# Data options
parser.add_argument('-data_dir', default='data/cub_c10', help='data directory.')
parser.add_argument('-ids_file', default='trainids.txt', help='file specifying which class labels are used for training. Can also be trainvalids.txt')
parser.add_argument('-batch_size', type=int, default=40, help='number of sequences to train on in parallel')
parser.add_argument('-image_dim', type=int, default=1024, help='image feature dimension')
parser.add_argument('-emb_dim', type=int, default=1536, help='embedding dimension')
parser.add_argument('-image_noop', type=int, default=1, help='if 1, the image encoder is a no-op. In this case emb_dim and image_dim must match.')
parser.add_argument('-randomize_pair', type=int, default=0, help='if 1, images and captions of the same class are randomly paired.')
parser.add_argument('-doc_length', type=int, default=201, help='document length')
parser.add_argument('-nclass', type=int, default=200, help='number of classes')
parser.add_argument('-dropout', type=float, default=0.0, help='dropout rate')
parser.add_argument('-gpuid', type=int, default=0, help='which GPU to use. -1 = use CPU')
parser.add_argument('-seed', type=int, default=123, help='torch manual random number generator seed')
parser.add_argument('-savefile', default='sje_hybrid', help='filename to autosave the checkpoint to. Will be inside checkpoint_dir/')
parser.add_argument('-checkpoint_dir', default='cv', help='output directory where checkpoints get written')
parser.add_argument('-init_from', default='', help='initialize network parameters from checkpoint at this path')
parser.add_argument('-max_epochs', type=int, default=300, help='number of full passes through the training data')
parser.add_argument('-grad_clip', type=int, default=5, help='clip gradients at this value')
parser.add_argument('-learning_rate', type=float, default=0.0004, help='learning rate')
parser.add_argument('-learning_rate_decay', type=float, default=0.98, help='learning rate decay')
parser.add_argument('-learning_rate_decay_after', type=int, default=1, help='in number of epochs, when to start decaying the learning rate')
parser.add_argument('-print_every', type=int, default=100, help='how many steps/minibatches between printing out the loss')
parser.add_argument('-eval_val_every', type=int, default=1000, help='every how many iterations should we evaluate on validation data?')
parser.add_argument('-symmetric', type=int, default=1, help='whether to use symmetric form of SJE')
parser.add_argument('-num_caption', type=int, default=5, help='number of captions per image to be used for training')
parser.add_argument('-image_dir', default='images_th3', help='image directory in data')
parser.add_argument('-flip', type=int, default=0, help='flip sentence')
parser.add_argument('-bidirectional', type=int, default=0, help='use bidirectional version')
parser.add_argument('-avg', type=int, default=0, help='whether to time-average hidden units')
parser.add_argument('-cnn_dim', type=int, default=256, help='char-cnn embedding dimension')

# Parse the command-line arguments
opt = parser.parse_args()

if opt.image_noop:
    opt.emb_dim = opt.image_dim

torch.manual_seed(opt.seed)
print(opt)

if opt.bidirectional == 1:
    FixedRNN = require('modules.BidirectionalRNN')
else:
    FixedRNN = require('modules.FixedRNN')
DocumentCNN = require('modules.HybridCNN')
ImageEncoder = require('modules.ImageEncoder')
MultimodalMinibatchLoader = require('util.MultimodalMinibatchLoaderCaption')
model_utils = require('util.model_utils')

# Initialize CUDA for training on the GPU and fallback to CPU gracefully
if opt.gpuid >= 0:
    import torch.cuda as cuda
    try:
        print('Using CUDA on GPU ' + str(opt.gpuid) + '...')
        cuda.set_device(opt.gpuid)
        cuda.manual_seed(opt.seed)
    except ImportError:
        print('CUDA is not available. Falling back to CPU mode.')
        opt.gpuid = -1  # Overwrite user setting

# Create the MultimodalMinibatchLoader
loader = MultimodalMinibatchLoader.create(
    opt.data_dir, opt.nclass, opt.image_dim, opt.doc_length,
    opt.batch_size, opt.randomize_pair, opt.ids_file, opt.num_caption,
    opt.image_dir, opt.flip)

# Create the checkpoint directory if it doesn't exist
if not os.path.exists(opt.checkpoint_dir):
    os.mkdir(opt.checkpoint_dir)

do_random_init = False
if len(opt.init_from) > 0:
    print('loading from checkpoint ' + opt.init_from)
    checkpoint = torch.load(opt.init_from)
    protos = checkpoint['protos']
else:
    protos = {}
    protos['enc_image'] = ImageEncoder.enc(opt.image_dim, opt.emb_dim, opt.image_noop)
    protos['enc_doc'] = DocumentCNN.cnn(loader.alphabet_size, opt.emb_dim, opt.dropout, opt.avg, opt.cnn_dim)
    protos['enc_image'].train()
    protos['enc_doc'].train()
    do_random_init = True

if opt.gpuid >= 0:
    for k, v in protos.items():
        if hasattr(v, 'weights') and v.weights is not None:
            v.weights = v.weights.float().cuda()
            v.grads = v.grads.float().cuda()
        else:
            v.cuda()
params, grad_params = model_utils.combine_all_parameters(protos['enc_image'], protos['enc_doc'])
if do_random_init:
    params.data.uniform_(-0.08, 0.08)  # small numbers uniform

acc_batch = 0.0
acc_smooth = 0.0

def JointEmbeddingLoss(fea_txt, fea_img, labels):
    batch_size = fea_img.size(0)
    num_class = fea_txt.size(0)
    score = torch.zeros(batch_size, num_class)
    txt_grads = fea_txt.clone().fill_(0)
    img_grads = fea_img.clone().fill_(0)

    loss = 0
    global acc_batch
    acc_batch = 0.0
    for i in range(batch_size):
        for j in range(num_class):
            score[i, j] = torch.dot(fea_img[i], fea_txt[j])
        label_score = score[i, labels[i]]
        for j in range(num_class):
            if j != labels[i]:
                cur_score = score[i, j]
                thresh = cur_score - label_score + 1
                if thresh > 0:
                    loss += thresh
                    txt_diff = fea_txt[j] - fea_txt[labels[i]]
                    img_grads[i].add_(txt_diff)
                    txt_grads[j].add_(fea_img[i])
                    txt_grads[labels[i]].add_(-fea_img[i])
        max_score, max_ix = torch.max(score[i].unsqueeze(0), 1)
        if max_ix.item() == labels[i]:
            acc_batch += 1

    acc_batch = 100 * (acc_batch / batch_size)
    denom = batch_size * num_class
    res = {1: txt_grads.div_(denom), 2: img_grads.div_(denom)}
    global acc_smooth
    acc_smooth = 0.99 * acc_smooth + 0.01 * acc_batch
    return loss / denom, res

def wrap_emb(inp, nh, nx, ny, labs):
    x = inp[:nh*nx].clone().reshape(nx, nh)
    y = inp[nh*nx:nh*nx + nh*ny].clone().reshape(ny, nh)
    loss, grads = JointEmbeddingLoss(x, y, labs)
    dx = grads[0]
    dy = grads[1]
    grad = torch.cat((dx.reshape(nh*nx), dy.reshape(nh*ny)))
    return loss, grad

if opt.checkgrad == 1:
    print('\nChecking embedding gradient\n')
    nh = 3
    nx = 4
    ny = 2
    txt = torch.randn(nx, nh)
    img = torch.randn(ny, nh)
    labs = torch.randperm(nx)
    initpars = torch.cat((txt.clone().reshape(nh*nx), img.clone().reshape(nh*ny)))
    opfunc = lambda curpars: wrap_emb(curpars, nh, nx, ny, labs)
    diff, dC, dC_est = checkgrad(opfunc, initpars, 1e-3)
    print(dC)
    print(dC_est)
    print(diff)
    debug.debug()

def feval_wrap(pars):
    txt, img, labels = loader.next_batch()
    return feval(pars, txt, img, labels)

def feval(newpars, txt, img, labels):
    if newpars is not params:
        params.copy_(newpars)
    grad_params.zero_()

    if opt.gpuid >= 0:
        txt = txt.float().cuda()
        img = img.float().cuda()
        labels = labels.float().cuda()
    
    fea_txt = protos.enc_doc(txt)
    fea_img = protos.enc_image(img)

    loss, grads = JointEmbeddingLoss(fea_txt, fea_img, labels)
    dtxt = grads[0]
    dimg = grads[1]

    if opt.symmetric == 1:
        loss2, grads2 = JointEmbeddingLoss(fea_img, fea_txt, labels)
        dtxt.add_(grads2[1])
        dimg.add_(grads2[0])
        loss += loss2
    
    protos.enc_doc.backward(txt, dtxt)
    protos.enc_image.backward(img, dimg)

    return loss, grad_params

train_losses = []
val_losses = []
optim_state = {"lr": opt.learning_rate, "alpha": opt.decay_rate}
iterations = opt.max_epochs * loader.ntrain
iterations_per_epoch = loader.ntrain
loss0 = None
for i in range(1, iterations + 1):
    epoch = i / loader.ntrain

    timer = torch.Timer()
    _, loss = optim.rmsprop(feval_wrap, params, **optim_state)
    time = timer.elapsed_time()

    train_loss = loss[0]  # the loss is inside a list, extract it
    train_losses.append(train_loss)

    # exponential learning rate decay
    if i % loader.ntrain == 0 and opt.learning_rate_decay < 1:
        if epoch >= opt.learning_rate_decay_after:
            decay_factor = opt.learning_rate_decay
            optim_state["lr"] *= decay_factor  # decay learning rate
            print(f"decayed learning rate by a factor {decay_factor} to {optim_state['lr']}")

    # every now and then or on last iteration
    if i % opt.eval_val_every == 0 or i == iterations:
        # evaluate loss on validation data
        val_loss = 0
        val_losses.append(val_loss)

        savefile = f"{opt.checkpoint_dir}/lm_{opt.savefile}_{opt.learning_rate:.5f}_{opt.symmetric}_{opt.num_caption}_{opt.ids_file}.t7"
        print("saving checkpoint to", savefile)
        checkpoint = {
            "protos": protos,
            "opt": opt,
            "train_losses": train_losses,
            "val_loss": val_loss,
            "val_losses": val_losses,
            "i": i,
            "epoch": epoch,
            "vocab": loader.vocab_mapping,
        }
        torch.save(checkpoint, savefile)

    if i % opt.print_every == 0:
        print(f"{i}/{iterations} (ep {epoch:.3f}), loss={train_loss:.2f}, acc1={acc_batch:.2f}, acc2={acc_smooth:.4f}, g/p={grad_params.norm() / params.norm():.4e}, t/b={time:.2f}s")

    if i % 10 == 0:
        torch.cuda.empty_cache()

    # handle early stopping if things are going really bad
    if loss0 is None:
        loss0 = loss[0]
    if loss[0] > loss0 * 3 and i > 10:
        print("loss is exploding, aborting.")
        break  # halt
