import argparse
import random

import dateutil.tz
import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader
import datetime

from torchvision.transforms import transforms

from modules.joint_embedding_trainer import JointEmbeddingTrainer
from util.multimodal_dataset import MultimodalDataset, AspectResize


def parse_args():
    parser = argparse.ArgumentParser(description='Train a Joint Embedding network')
    # required
    parser.add_argument('--name', dest='name', type=str, required=True)
    parser.add_argument('--data-dir', dest='data_dir', type=str, required=True)
    parser.add_argument('--class-name-file', dest='class_file', type=str, required=True,
                        help="Path to text file that contains class names, each line")
    parser.add_argument('--label-csv', dest='label_csv', type=str, required=True,
                        help="A csv file that contains filename and label name ")
    # optional
    parser.add_argument('--snapshot-interval', dest='snapshot_interval', type=int, default=5)
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=4)
    parser.add_argument('--emd-dim', dest='emb_dim', type=int, default=1536, help="Joint embedding dimension")
    parser.add_argument('--imgsize', dest='imgsize', type=int, default=256, help="Image input size")
    parser.add_argument('--doc-length', dest='doc_length', type=int, default=170, help="Text input size")
    parser.add_argument('--dropout', dest='dropout', type=float, default=0.0, help="Dropout for CNN-RNN")
    parser.add_argument('--epoch-start', dest='epoch_start', type=int, default=1, help="Epoch start")
    parser.add_argument('--max-epoch', dest='max_epoch', type=int, default=2, help="Number of epoch")
    parser.add_argument('--workers', dest='workers', type=int, default=4,
                        help="Number of worker required during data loading")
    parser.add_argument('--drop-last', dest='drop_last', default=False, action='store_true',
                        help="Drop the last batch if number of remaining items are less than the batch size")
    parser.add_argument('--test', dest='test_phase', default=False, action='store_true',
                        help="Specify enable test phase. Default will train")
    parser.add_argument('--finetune', dest='finetune', default=False, action='store_true',
                        help="Specify enable test phase. Default will train")
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
    torch.cuda.manual_seed_all(args.manualSeed)
    # load classnames
    with open(args.class_file) as fp:
        args.classnames = list(map(lambda x: x.strip(), fp.readlines()))
    
    # prepare GPU and batches
    s_gpus = args.gpu_id.split(',')
    gpus = [int(ix) for ix in s_gpus]
    num_gpus = len(gpus)
    batch_size = args.batch_size * num_gpus
    
    # image transform
    image_transform = transforms.Compose([
        AspectResize(args.imgsize),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    # load dataset
    train_dataset = MultimodalDataset(args.data_dir, args.classnames, args.label_csv, args.doc_length, args.imgsize,
                                      split=phase,
                                      image_transform=image_transform)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, drop_last=args.drop_last,
                                  shuffle=True, num_workers=int(args.workers))
    
    trainer = JointEmbeddingTrainer(output_dir, args)
    trainer.train(train_dataloader, args)


if __name__ == '__main__':
    main()
