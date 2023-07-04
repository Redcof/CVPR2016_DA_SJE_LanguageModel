from abc import abstractmethod

import torch
from torchvision.transforms import transforms


class EmbeddingFactory:
    EMBEDDING_STRATEGIES = ('char', 'word', 'word2vec', 'glove25', 'glove50', 'glove100', 'glove200', 'glove300',
                            'fasttext-en', 'fasttext')
    
    def __init__(self, strategy, doc_length, **kwargs):
        assert strategy in EmbeddingFactory.EMBEDDING_STRATEGIES, "The allowed strategies are: {} but {} given".format(
            EmbeddingFactory.EMBEDDING_STRATEGIES, strategy
        )
        if strategy == "char":
            txt_embedding = CharEmbedTransform(doc_length)
        elif strategy == "word":
            txt_embedding = WordEmbedTransform(doc_length, **kwargs)
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
            txt_embedding = FasttestTrainTransform(**kwargs)
        elif strategy == "fasttext-en":
            txt_embedding = FasttextPretrainedTransform(**kwargs)
        elif strategy == "openai":
            txt_embedding = OpenAIGPTTransform(**kwargs)
        self.txt_embedding_transform = txt_embedding


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
    
    def __init__(self, doc_length, captions):
        words = []
        for caption in captions:
            words.extend(caption.split())
        words = list(set(words))
        super().__init__(words, doc_length)
    
    def split(self, caption):
        return caption.split()


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
