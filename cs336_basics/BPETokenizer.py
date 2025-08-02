import os
from io import BytesIO
import regex as re
import logging

from collections import defaultdict

from multiprocessing import Queue, Process
from concurrent.futures import ProcessPoolExecutor, Future

from typing_extensions import List, Iterable, Set, BinaryIO, Dict, Tuple, Sequence, Iterator

from .pretokenization import pretokenize

class BPETokenizer:
    def __init__(self,
                 vocab: dict[int, bytes],
                 merges: list[tuple[bytes, bytes]],
                 speical_tokens: Sequence[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = speical_tokens or []

    def encode(self, text: str) -> List[int]:
        byte_tokens = list(pretokenize(text, self.special_tokens))
        byte_special_tokens = {s.encode('utf-8') for s in self.special_tokens}
        reversed_vocab = {v: k for k, v in self.vocab.items()}
        tokens_list: List[List[int]] = []
        for byte_token in byte_tokens:
            tokens: List[int] = []
            if byte_token in byte_special_tokens:
                tokens.append(reversed_vocab[byte_token])
            else:
                tokens.extend(
                    map(lambda x: reversed_vocab[bytes([x])], byte_token)
                )
            tokens_list.append(tokens)
        
        for i in range(len(tokens_list)):
            if len(tokens_list[i]) == 1: continue
            token = tokens_list[i]
            for pair in self.merges:
                j = 0
                new_tokens = []
                merged_token = reversed_vocab[pair[0] + pair[1]]
                while j < len(token):
                    if j < len(token) - 1 and (self.vocab[token[j]], self.vocab[token[j + 1]]) == pair:
                        new_tokens.append(merged_token)
                        j += 2
                    else:
                        new_tokens.append(token[j])
                        j += 1
                token = new_tokens
            tokens_list[i] = token
        
        ret = [token for tokens in tokens_list for token in tokens]
        return ret

    def decode(self, token_ids: List[int]) -> str:
        replace_char = "\uFFFD".encode('utf-8')
        vocab_size = len(self.vocab)
        byte_tokens = bytes()
        for token in token_ids:
            byte_token = self.vocab[token] if token < vocab_size else replace_char
            byte_tokens += byte_token
        
        return byte_tokens.decode('utf-8', errors='ignore')
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for s in iterable:
            for token in self.encode(s):
                yield token
    
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        pass
                
