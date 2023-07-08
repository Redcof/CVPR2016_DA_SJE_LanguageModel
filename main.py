import argparse
import os
import pathlib
import random

import dateutil.tz
import torch
from git import Repo
from torch.utils.data import DataLoader
import datetime

from torchvision.transforms import transforms

from modules.joint_embedding_trainer import JointEmbeddingTrainer
from util.aspect_resize_transform import AspectResize
from util.multimodal_dataset import MultimodalDataset
from util.text_encoder_interface import EmbeddingFactory


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
    parser.add_argument('--embedding-strategy', dest='embedding_strategy', type=str, default='char',
                        choices=('char', 'word', 'word2vec', 'glove25', 'glove50', 'glove100', 'glove200', 'glove300',
                                 'fasttext-en', 'fasttext'),
                        help="Specify the initial text encoding strategy. "
                             "char: one-hot-encode each character."
                             "word: one-hot-encode each word. "
                             "word2vec: encode word with word2vec encoder."
                             "glove: encode word with GloVe encoder. Specify 25, 50, 100, 200, and 300 as per the need."
                             "fasttext-en: use pretrained FastText English model."
                             "fasttext: First train the given caption data and use it for encoding."
                             "openai: Use openai pretrained document encoder."
                        )
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
    parser.add_argument('--predict', dest='predict', default=False, action='store_true',
                        help="Specify to generate text embedding")
    parser.add_argument('--bulk', dest='bulk', default=0, type=int,
                        help="Used during inference. Caption file contains multiple captions. This value indicates "
                             "if we want to generate embedding as a bulk per file or not. Specifying less than 2 "
                             "will be considered as single caption.")
    parser.add_argument('--vocabulary-txt', dest='vocabulary_txt', type=str, help="A file that contains the vocabulary")
    parser.add_argument('--NET-IMG', dest='NET_IMG', default='', help="Path to Image Encoder Network")
    parser.add_argument('--NET-TXT', dest='NET_TXT', default='', help="Path to Text Encoder Network")
    parser.add_argument('--cuda', dest='cuda', default=False, action='store_true',
                        help="To enable training on GPU")
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
    phase = "test" if args.test_phase else ("predict" if args.predict else "train")
    # prepare output directory
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    args.timestamp = timestamp  # save timestamp in the config
    project_root = pathlib.Path(__file__).parents[0]
    repo = Repo(project_root)
    args.git_checksum = repo.git.rev_parse("HEAD")  # save commit checksum
    output_dir = 'output/%s_%s_%s_%s' % (args.name, args.embedding_strategy, phase, timestamp)
    print("Output:", output_dir)
    # set random seeds
    if args.manualSeed is None:
        args.manualSeed = 47
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
    
    if phase in ("train", "test"):
        # read all the captions
        captions = []
        dataset_path = pathlib.Path(args.data_dir)
        caption_dir = dataset_path / phase / "captions"
        
        # create a caption initial encoder object
        embed_transform = EmbeddingFactory(args, caption_dir).get()
        # set vocab length
        args.vocab_length = embed_transform.vocab_length
        args.vocabulary = embed_transform.vocabulary
        caption_transform = transforms.Compose([
            embed_transform,
        ])
        # image transform
        image_transform = transforms.Compose([
            AspectResize(args.imgsize),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        # load dataset
        train_dataset = MultimodalDataset(args.data_dir, args.classnames, args.label_csv, args.doc_length, args.imgsize,
                                          split=phase,
                                          image_transform=image_transform,
                                          caption_transform=caption_transform)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, drop_last=args.drop_last,
                                      shuffle=True, num_workers=int(args.workers))
        
        trainer = JointEmbeddingTrainer(output_dir, args)
        trainer.train(train_dataloader, args)
    elif phase == "predict":
        trainer = JointEmbeddingTrainer(output_dir, args)
        trainer.predict(args)


if __name__ == '__main__':
    main()
