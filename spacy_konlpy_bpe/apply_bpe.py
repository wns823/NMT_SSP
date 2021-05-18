import os
import argparse

from tqdm import tqdm
from tokenizers import SentencePieceBPETokenizer
import multiprocessing as mp


def define_argparser():

    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--bpe_path', type=str, required=True, help='BPE file path')
    parser.add_argument('-i', '--input_path', type=str, required=True, help='Input file path')
    parser.add_argument('-o', '--output_path', type=str, required=True, help='Input file path')
    parser.add_argument('--threads', type=int, default=-1, help='Num threads')
    # parser.add_argument('-mc', '--monolingual_corpus', action="store_true", help='Preprocess monolingual data')       

    args = parser.parse_args()
    return args
    
def bpe_encode(text):
    return tokenizer.encode(text).tokens



if __name__ == '__main__':
    args = define_argparser()

    bpe_path = args.bpe_path
    input_path = args.input_path
    output_path = args.output_path
    if args.threads == -1:
        num_process = mp.cpu_count()
    else:
        num_process = args.threads    


    with open(input_path, 'r', encoding='utf8') as f:
        corpus = f.read().split('\n')
    corpus = [line for line in corpus if len(line)>0]

    bpe_file_vocab = f'{bpe_path}-vocab.json'
    bpe_file_merges = f'{bpe_path}-merges.txt'
    if os.path.exists(bpe_file_vocab) and os.path.exists(bpe_file_merges):
        tokenizer = SentencePieceBPETokenizer(bpe_file_vocab, bpe_file_merges)
    else:
        raise 'No BPE available'


    #### Saving files
    pool = mp.Pool(processes=num_process)
    processed_corpus = pool.map(bpe_encode, corpus)
    pool.close()
    pool.join()

    with open(output_path, 'w', encoding='utf8') as f:
        for line in tqdm(processed_corpus):
            line = ' '.join(line)
            f.write(str(line) + '\n')

