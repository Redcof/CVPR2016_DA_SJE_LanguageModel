import argparse
import random
import time

import dateutil.tz
import torch
from torch import optim, autograd, nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import datetime

from torchvision import transforms

from python.modules.HybridCNN import HybridCNN
from python.modules.ImageEncoder import ImageEncoder
from python.util.model_utils import weights_init
from python.util.multimodal_dataset import MultimodalDataset, AspectResize


def parse_args():
    parser = argparse.ArgumentParser(description='Train a Joint Embedding network')
    parser.add_argument('--name', dest='name', type=str, required=True)
    parser.add_argument('--data-dir', dest='data_dir', type=str, required=True)
    parser.add_argument('--class-name-file', dest='class_file', type=str, required=True,
                        help="Path to text file that contains class names, each line")
    parser.add_argument('--batch-size', dest='batch_size', type=str, default=16)
    parser.add_argument('--emd-dim', dest='emb_dim', type=str, default=1536, help="Joint embedding dimension")
    parser.add_argument('--imgsize', dest='imgsize', type=int, default=256, help="Image input size")
    parser.add_argument('--doc-length', dest='doc_length', type=int, default=100, help="Text input size")
    parser.add_argument('--dropout', dest='imgsize', type=float, default=0.0, help="Dropout for CNN-RNN")
    parser.add_argument('--epoch-start', dest='epoch_start', type=int, default=1, help="Epoch start")
    parser.add_argument('--epoch', dest='epoch', type=int, default=100, help="Number of epoch")
    parser.add_argument('--workers', dest='workers', type=int, default=4,
                        help="Number of worker required during data loading")
    parser.add_argument('--drop-last', dest='drop_last', default=False, action='store_true',
                        help="Drop the last batch if number of remaining items are less than the batch size")
    parser.add_argument('--test-phase', dest='test_phase', default=False, action='store_true',
                        help="Specify train or test phase")
    parser.add_argument('--cuda', dest='cuda', default=False, action='store_true',
                        help="To enable training on GPU")
    parser.add_argument('--NET-IMG', dest='NET_IMG', default='', help="Path to Image Encoder Network")
    parser.add_argument('--NET-TXT', dest='NET_TXT', default='', help="Path to Text Encoder Network")
    parser.add_argument('--gpu', dest='gpu_id', type=str, default='0')
    parser.add_argument('--manual-seed', dest='manualSeed', type=int, help='manual seed', default=47)
    parser.add_argument('--lr-img', dest="lr_img", type=float, default=0.0004, help='learning rate')
    parser.add_argument('--lr-txt', dest="lr_txt", type=float, default=0.0004, help='learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.98, help='learning rate decay')
    parser.add_argument('--lr_decay_after', type=int, default=1,
                        help='in number of epochs, when to start decaying the learning rate')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    phase = "test" if args.test_phase else "train"
    num_gpu = 1
    # prepare output directory
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = 'output/%s_%s_%s' % (args.name, phase, timestamp)
    print("Output:", output_dir)
    # set random seeds
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if args.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)
    # load classnames
    with open(args.class_file) as fp:
        args.classnames = fp.readlines()
    
    # prepare GPU and batches
    s_gpus = args.gpu_id.split(',')
    gpus = [int(ix) for ix in s_gpus]
    num_gpus = len(gpus)
    batch_size = args.batch_size * num_gpus
    torch.cuda.set_device(gpus[0])
    cudnn.benchmark = True
    
    # image transform
    image_transform = transforms.Compose([
        AspectResize(args.imgsize),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    # load dataset
    train_dataset = MultimodalDataset(args.data_dir, args.classnames, args.doc_length, args.imgsize, split=phase,
                                      image_transform=image_transform)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, drop_last=args.drop_last,
                                  shuffle=True, num_workers=int(args.workers))
    # create models
    netTXT = HybridCNN(train_dataset.vocab_length, args.emb_dim, dropout=args.dropout)
    dim_googlelenet_output_dim = 1000
    netIMG = ImageEncoder(dim_googlelenet_output_dim, args.emb_dim)
    
    # Network weight initialization
    netTXT.avg(weights_init)
    netIMG.avg(weights_init)
    
    # move models to CUDA
    if args.cuda:
        netIMG.cuda()
        netTXT.cuda()
    
    # initialize training parameters
    generator_lr = args.lr_img
    discriminator_lr = args.lr_txt
    lr_decay_step = args.lr_decay
    optimizerG = optim.RMSprop(netTXT.get_learnable_params(), lr=args.lr)
    optimizerD = optim.RMSprop(netTXT.get_learnable_params(), lr=args.lr)
    
    # train loop
    for epoch_idx in range(args.epoch_start, args.epoch + 1):
        start_t = time.time()
        #################
        # train one epoch
        for batch_idx, (txt, img, label) in enumerate(train_dataloader, 0):
            txt = autograd.Variable(txt)
            img = autograd.Variable(img)
            label = autograd.Variable(label)
            if args.cuda:
                txt.cuda()
                img.cuda()
                label.cuda()
            
            inp_img, enc_img, mu_img, logvar_img = nn.parallel.data_parallel(netIMG, img, gpus)
            einp_txt, enc_txt, mu_txt, logvar_txt = nn.parallel.data_parallel(netTXT, txt, gpus)
        #################


if __name__ == '__main__':
    main()
