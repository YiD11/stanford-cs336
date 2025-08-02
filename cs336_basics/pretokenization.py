import os
from io import BytesIO
import regex as re
import logging

from collections import defaultdict

from multiprocessing import Queue, Process
from concurrent.futures import ProcessPoolExecutor, Future

from typing_extensions import List, Iterable, Set, BinaryIO, Dict, Tuple, Sequence

text = "Hello, how <|endoftext|><|endoftext|> are you?<|endoftext|>"


def split_by_special_tokens(
        s: str, special_tokens: list[str], drop_special: bool = False
) -> List[str]:
    """
    Split a string by special tokens, optionally dropping them.
    """
    if not special_tokens:
        return [s]
    special_tokens = sorted(special_tokens, key=len, reverse=True)

    pattern = r'|'.join(re.escape(token) for token in special_tokens)
    if not drop_special:
        pattern = f"({pattern})"
    ret = re.split(pattern, s)
    return ret

def pretokenize(s: str, special_tokens: Sequence[str] | str, drop_special: bool = False) -> Iterable[bytes]:
    if isinstance(special_tokens, str):
        special_tokens = {special_tokens}

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    parts = split_by_special_tokens(s, special_tokens, drop_special=drop_special)
    for part in parts:
        if part in special_tokens:
            yield part.encode("utf-8")
        else:
            for sub in re.finditer(PAT, part):
                yield sub.group().encode("utf-8")

def split_by_special_tokensv2(text: str, special_tokens: list[str]) -> list[str]:
    """
    Split on the special tokens
    example: 
        text = "Hello world! <|endoftext|> Great!" 
        special_tokens = "<|endoftext|>"
        result = ['Hello world! ', '<|endoftext|>', ' Great!']
    """
    special_tokens_sorted = sorted(special_tokens, key=lambda x: -len(x))
    if not special_tokens_sorted:
        parts = [text]
    else:
        pattern = "|".join(re.escape(tok) for tok in special_tokens_sorted)
        parts = re.split('(' + pattern + ')', text)

    return parts

# def pretokenize(text: str, special_tokens: list[str], drop_special_token: bool = True) -> list[bytes]:
#     """
#     Seperating text into pretokens
#     Special tokens are independent pretokens
#     """
#     parts = split_by_special_tokens(text, special_tokens)

#     PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
#     tokens_list = []
#     for part in parts:
#         if part in special_tokens:
#             if not drop_special_token:  # Keep special tokens, otherwise ignore
#                 spec_tok_bytes = part.encode('utf-8')
#                 tokens_list.append([spec_tok_bytes])
#         else:
#             str_tokens = re.findall(PAT, part)
#             part_tokens = [s.encode('utf-8') for s in str_tokens]
#             tokens_list.append(part_tokens)
#     tokens = [token for part_tokens in tokens_list for token in part_tokens]
#     return tokens


def find_special_tokens(fp: BinaryIO, file_size: int, chunk_size: int, special_token: bytes) -> List[int]:
    '''
    guarantee that special_token only appears at the end of a chunk
    '''
    ret = []
    while fp.tell() < file_size:
        pos = fp.tell()
        end = min(pos + chunk_size + len(special_token) - 1, file_size)
        chunk = fp.read(end - pos)
        idx = chunk.find(special_token)
        if idx == -1:
            fp.seek(pos + chunk_size)
            continue

        speical_token_end = pos + idx + len(special_token)
        ret.append(speical_token_end)
        fp.seek(speical_token_end)
    return ret

def preprocess(s: str, special_tokens: list[str]):
    if s is None or len(s.strip()) == 0:
        return [], {}, {}
    chunks = list(pretokenize(s, special_tokens, drop_special=True))
    return chunks

def run_train_bpe(
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str],
        **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    futures: List[Future] = []
    chunks: List[bytes] = []
    count: Dict[Tuple[int, int], int] = defaultdict(int)
    pair_to_indices: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
    print("chunking")
        
    with ProcessPoolExecutor(max_workers=8) as executor:
        with open(input_path, 'rb') as fp:
            fp.seek(0)
            file_size = fp.seek(0, os.SEEK_END)
            fp.seek(0)
            print("finding special tokens")
            ends = find_special_tokens(fp, file_size, min(4096, file_size), "<|endoftext|>".encode('utf-8'))
            split_indices = [0] + ends + [file_size]

            print("pretokenizing")
            for i, (start, end) in enumerate(zip(split_indices[:-1], split_indices[1:])):
                fp.seek(start)
                sub_chunk = fp.read(end - start).decode("utf-8", errors="ignore")
                futures.append(executor.submit(preprocess, sub_chunk, special_tokens))
        
    print("counting pairs")
    for i, f in enumerate(futures):
        sub_chunk = f.result()
        chunks.extend(sub_chunk)

    for i, sub_chunk in enumerate(chunks):
        for j1, j2 in zip(sub_chunk, sub_chunk[1:]):
            count[j1, j2] += 1
            pair_to_indices[j1, j2].add(i)

    merges: List[Tuple[bytes, bytes]] = []
    vocabs: Dict[int, bytes] = {i: bytes([i]) for i in range(256)} | {256 + i: chunk.encode('utf-8') for i, chunk in enumerate(special_tokens)}
    num_merges = vocab_size - len(vocabs)

    print("merging pairs")
    for iter in range(num_merges):

        pair, _ = max(
            count.items(),
            key=lambda x: (x[1], vocabs[x[0][0]].decode('utf-8',errors='ignore'), vocabs[x[0][1]].decode('utf-8',errors='ignore'))
        )
        
        is_merged = False
        next_token = len(vocabs)
        vocabs[next_token] = vocabs[pair[0]] + vocabs[pair[1]]
        merges.append((vocabs[pair[0]], vocabs[pair[1]]))

        for i in pair_to_indices[pair]:
            sub_chunk = chunks[i]
            new_chunk: List[int] = []
            positions: List[int] = []
            pos = j = 0
            while j < len(sub_chunk):
                if j < len(sub_chunk) - 1 and (sub_chunk[j], sub_chunk[j + 1]) == pair:
                    new_chunk.append(next_token)
                    is_merged = True
                    j += 2
                    positions.append(pos)
                else:
                    new_chunk.append(sub_chunk[j])
                    j += 1
                pos += 1
            chunks[i] = new_chunk

            for pos in positions:
                count[pair] -= 1

                if pos > 0:
                    if new_chunk[pos - 1] == next_token:
                        pass
                        # count[pair[1], pair[0]] -= 1
                    else:
                        count[new_chunk[pos - 1], pair[0]] -= 1
                    count[new_chunk[pos - 1], new_chunk[pos]] += 1
                    pair_to_indices[new_chunk[pos - 1], new_chunk[pos]].add(i)
                
                if pos < len(new_chunk) - 1:
                    if new_chunk[pos + 1] == next_token:
                        count[pair[1], pair[0]] -= 1
                    else:
                        count[pair[1], new_chunk[pos + 1]] -= 1
                    count[new_chunk[pos], new_chunk[pos + 1]] += 1
                    pair_to_indices[new_chunk[pos], new_chunk[pos + 1]].add(i)

        pair_to_indices.pop(pair, None)
        # if not is_merged: break

    return (vocabs, merges)

if __name__ == "__main__":
    pass