import os
from abc import abstractmethod

import torch
from torchvision.transforms import transforms

from util.model_utils import clean


class EmbeddingFactory:
    EMBEDDING_STRATEGIES = ('char', 'word', 'word2vec', 'glove25', 'glove50', 'glove100', 'glove200', 'glove300',
                            'fasttext-en', 'fasttext')
    
    def __init__(self, args, caption_dir, **kwargs):
        strategy, doc_length = args.embedding_strategy, args.doc_length
        assert strategy in EmbeddingFactory.EMBEDDING_STRATEGIES, "The allowed strategies are: {} but {} given".format(
            EmbeddingFactory.EMBEDDING_STRATEGIES, strategy
        )
        captions = []
        if args.embedding_strategy in ('word', 'fasttext'):
            caption_files = [os.path.join(caption_dir, file) for file in os.listdir(caption_dir) if
                             file.endswith(".txt")]
            for caption_file in caption_files:
                with open(caption_file) as fp:
                    captions.extend([clean(caption.lower().strip()) for caption in fp.readlines()])
            
            max_char_len = float("-Inf")
            max_word_len = float("-Inf")
            for caption in captions:
                if len(caption) > max_char_len:
                    max_char_len = len(caption)
                if len(caption.strip().split()) > max_word_len:
                    max_word_len = len(caption.strip().split())
            print("Doc length(char):", max_char_len)
            print("Doc length(word):", max_word_len)
        if args.predict and args.embedding_strategy == "word":
            assert os.path.exists(args.vocabulary_txt), "Inference using 'word' embedding " \
                                                        "strategy requires a `vocabulary.txt` file " \
                                                        "generated during training"
            with open(args.vocabulary_txt) as fp:
                vocabulary = list(map(lambda x: x.strip(), fp.readlines()))
        else:
            vocabulary = []
            for caption in captions:
                vocabulary.extend(caption.split())
            vocabulary = list(set(vocabulary))
        if strategy == "word":
            txt_embedding = WordEmbedTransform(doc_length, vocabulary)
        elif strategy == "word2vec":
            txt_embedding = Word2VecTransform(**kwargs)
        elif strategy == "glove25":
            txt_embedding = GloVeEmbeddingTransform(25)
        elif strategy == "glove50":
            txt_embedding = GloVeEmbeddingTransform(50)
        elif strategy == "glove100":
            txt_embedding = GloVeEmbeddingTransform(100)
        elif strategy == "glove200":
            txt_embedding = GloVeEmbeddingTransform(200)
        elif strategy == "glove300":
            txt_embedding = GloVeEmbeddingTransform(300)
        elif strategy == "fasttext":
            txt_embedding = FasttestTrainTransform(captions, args.emb_dim)
        elif strategy == "fasttext-en":
            txt_embedding = FasttextPretrainedTransform(**kwargs)
        elif strategy == "openai":
            txt_embedding = OpenAIGPTTransform(**kwargs)
        else:
            # default char embedding transform
            txt_embedding = CharEmbedTransform(doc_length)
        self.txt_embedding_transform = txt_embedding
    
    def get(self):
        return self.txt_embedding_transform


class TextEncoderTransformInterface(torch.nn.Module):
    
    def __init__(self, vocabulary, doc_length):
        super().__init__()
        self.doc_length = doc_length
        self.vocabulary = vocabulary
        self.vocab_length = len(self.vocabulary)
        MINIMUM = 70
        if self.doc_length < MINIMUM:
            self.doc_length = MINIMUM
    
    @abstractmethod
    def split(self, caption):
        """
        Preprocess the caption and create a list of items to be encoded.
        For instance, for 'char-encoding' it must create a list of
        characters form the given sentence.
        :param caption: The given caption
        :return: A list
        """
        ...
    
    def embed_index(self, item):
        """
        A default method to return item(word or char) index in the given caption
        :param item:  str
        :return: Integer
        """
        try:
            return self.vocabulary.index(item)
        except:
            return 0
    
    def forward(self, caption):
        """Must remove '\n' character. It currently supports ASCII characters"""
        # lowercase
        caption = caption.lower().strip()
        # split into smallest quantities as per strategy
        char_list = self.split(caption)
        
        # truncate if the items if large
        if len(char_list) > self.doc_length:
            char_list = char_list[:self.doc_length]
        
        # encode characters
        index_list = [self.embed_index(item) for item in char_list]
        
        # place encoding into a tensor
        embedding_tensor = torch.zeros(self.doc_length, self.vocab_length)
        
        for idx, item_idx in enumerate(index_list[1:]):
            try:
                embedding_tensor[idx, item_idx] = 1
            except IndexError as e:
                print("Error: Caption:", caption, len(caption), idx, item_idx, len(index_list), embedding_tensor.shape)
                raise e
        return embedding_tensor


class CharEmbedTransform(TextEncoderTransformInterface):
    
    def __init__(self, doc_length):
        vocabulary = "\nabcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "
        super().__init__(vocabulary, doc_length)
    
    def split(self, caption):
        char_list = []
        for word in caption.split():
            char_list.append(' ')
            for char in list(word):
                char_list.append(char)
        return char_list[1:]


class WordEmbedTransform(TextEncoderTransformInterface):
    
    def __init__(self, doc_length, vocabulary):
        super().__init__(vocabulary, doc_length)
    
    def split(self, caption):
        return clean(caption).split()


class Word2VecTransform(torch.nn.Module):
    
    def forward(self, caption):
        ...


class GloVeEmbeddingTransform(torch.nn.Module):
    
    def __init__(self, emb_dim):
        super().__init__()
        self.dim = emb_dim
    
    def forward(self, caption):
        ...


class FasttextPretrainedTransform(torch.nn.Module):
    
    def forward(self, caption):
        ...


class FasttestTrainTransform(torch.nn.Module):
    
    def __init__(self, all_captions, emb_dim):
        super().__init__()
        ...
    
    def forward(self, caption):
        ...


class OpenAIGPTTransform(torch.nn.Module):
    def forward(self, caption):
        ...
