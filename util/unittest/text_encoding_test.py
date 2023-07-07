import unittest

import torch

from util.multimodal_dataset import MultimodalDataset
from util.text_encoder_interface import CharEmbedTransform


class MyTestCase(unittest.TestCase):
    def test_something(self):
        sentence = "A quick call"
        doc_length = 350
        char_embedding_transform = CharEmbedTransform(doc_length)
        char_encoder = MultimodalDataset._basic_char_encoder(doc_length, lambda x, y: None)
        encoding1 = char_embedding_transform(sentence)
        encoding2 = char_encoder(sentence)
        
        self.assertEqual(torch.all(torch.eq(encoding2, encoding1)).item(), True)
        ...


if __name__ == '__main__':
    unittest.main()
