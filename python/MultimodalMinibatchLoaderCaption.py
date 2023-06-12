import io
import os
import torch

class MultimodalMinibatchLoaderCaption:
    def __init__(self, data_dir, nclass, img_dim, doc_length, batch_size,
                 randomize_pair, ids_file, num_caption, image_dir=None, flip=None):
        self.alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "
        self.dict = {char: i+1 for i, char in enumerate(self.alphabet)}
        self.alphabet_size = len(self.alphabet)

        self.files = []
        with open(os.path.join(data_dir, 'manifest.txt'), 'r') as file:
            for line in file:
                self.files.append(line.strip())

        self.trainids = []
        with open(os.path.join(data_dir, ids_file), 'r') as file:
            for line in file:
                self.trainids.append(line.strip())
        self.nclass_train = len(self.trainids)
        self.trainids_tensor = torch.zeros(len(self.trainids))
        for i, trainid in enumerate(self.trainids):
            self.trainids_tensor[i] = int(trainid)

        self.nclass = nclass
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.nclass = nclass
        self.img_dim = img_dim
        self.doc_length = doc_length
        self.ntrain = self.nclass_train
        self.randomize_pair = randomize_pair
        self.num_caption = num_caption
        self.image_dir = image_dir or ''
        self.flip = flip or 0

        torch.cuda.empty_cache()

    @staticmethod
    def create(data_dir, nclass, img_dim, doc_length, batch_size,
               randomize_pair, ids_file, num_caption, image_dir=None, flip=None):
        return MultimodalMinibatchLoaderCaption(data_dir, nclass, img_dim, doc_length,
                                                batch_size, randomize_pair, ids_file,
                                                num_caption, image_dir, flip)


    def next_batch(self):
        sample_ix = torch.randperm(self.nclass_train)[:self.batch_size]

        txt = torch.zeros(self.batch_size, self.doc_length, self.alphabet_size)
        img = torch.zeros(self.batch_size, self.img_dim)
        lab = torch.zeros(self.batch_size)

        for i in range(self.batch_size):
            id = int(self.trainids_tensor[sample_ix[i]])
            fname = self.files[id]

            if self.image_dir == '' or self.image_dir is None:
                cls_imgs = torch.load(os.path.join(self.data_dir, 'images', fname))
            else:
                cls_imgs = torch.load(os.path.join(self.data_dir, self.image_dir, fname))

            cls_sens = torch.load(os.path.join(self.data_dir, f'text_c{self.num_caption}', fname))

            sen_ix = torch.randint(1, cls_sens.size(3), (1,))
            ix = torch.randint(0, cls_sens.size(1), (1,))
            ix_view = torch.randint(0, cls_imgs.size(3), (1,))

            img[i] = cls_imgs[ix, :, ix_view]
            lab[i] = i + 1

            for j in range(cls_sens.size(2)):
                on_ix = cls_sens[ix, j, sen_ix]
                if on_ix == 0:
                    break
                if torch.rand(1) < self.flip:
                    txt[i, cls_sens.size(2) - j - 1, on_ix] = 1
                else:
                    txt[i, j, on_ix] = 1

        return txt, img, lab
'''
@staticmethod
def create(data_dir, nclass, img_dim, doc_length, batch_size,
            randomize_pair, ids_file, num_caption, image_dir=None, flip=None):
    return MultimodalMinibatchLoaderCaption(data_dir, nclass, img_dim, doc_length,
                                            batch_size, randomize_pair, ids_file,
                                            num_caption, image_dir, flip)
''''