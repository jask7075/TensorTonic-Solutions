import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        # YOUR CODE HERE
        self.word_to_id[self.pad_token] = 0
        self.word_to_id[self.unk_token] = 1
        self.word_to_id[self.bos_token] = 2
        self.word_to_id[self.eos_token] = 3

        counter = 4

        vocab = [word for string in texts for word in string.split()]
        vocab.sort()
        for word in vocab:
            if word not in self.word_to_id:
                self.word_to_id[word] = counter
                counter += 1

        self.id_to_word = {key:val for val, key in self.word_to_id.items()}

        print(self.word_to_id)
        print(self.id_to_word)
        self.vocab_size = counter
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        # YOUR CODE HERE
        word_list = [word.lower() for word in text.split()]
        word_list.sort()
        output_list = []
        for word in word_list:
            if word not in self.word_to_id:
                output_list.append(self.word_to_id[self.unk_token])
            else:
                output_list.append(self.word_to_id[word])
                
        return output_list
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        # YOUR CODE HERE
        output_str = ""
        len_ids = len(ids)
        for i in range(len_ids):
            if ids[i] in self.id_to_word:
                output_str += self.id_to_word[ids[i]]
            else:
                output_str += self.id_to_word[1]

            if i < len_ids - 1:
                    output_str += " "
                
        return output_str
