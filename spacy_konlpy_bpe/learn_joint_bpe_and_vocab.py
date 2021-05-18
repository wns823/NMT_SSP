import os
import argparse

from tqdm import tqdm
from tokenizers import SentencePieceBPETokenizer
import multiprocessing as mp


def define_argparser():

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--inputs', type=str, required=True, nargs=2, help='File paths')
    parser.add_argument('-o', '--outputs', type=str, required=True, help='Output file path')
    parser.add_argument('-s', '--bpe_iteration', type=int, required=True, default=32000, help='BPE iterations')
    parser.add_argument('--threads', type=int, default=-1, help='Num threads')
    # parser.add_argument('-mc', '--monolingual_corpus', action="store_true", help='Preprocess monolingual data')       

    args = parser.parse_args()
    return args
    


if __name__ == '__main__':
    args = define_argparser()

    print('Building BPE...')

    input_files = args.inputs
    output_file = args.outputs
    bpe_iteration = args.bpe_iteration
    if args.threads == -1:
        num_process = mp.cpu_count()
    else:
        num_process = args.threads    


    with open(input_files[0], 'r', encoding='utf8') as f:
        corpus_src = f.read().split('\n')
    corpus_src = [line for line in corpus_src if len(line)>0]
    with open(input_files[1], 'r', encoding='utf8') as f:
        corpus_trg = f.read().split('\n')
    corpus_trg = [line for line in corpus_trg if len(line)>0]



    bpe_dir = '/'.join(output_file.split('/')[:-1])
    bpe_file = output_file.split('/')[-1]

    tokenizer = SentencePieceBPETokenizer()
    tokenizer.train(files=input_files, vocab_size=bpe_iteration)
    tokenizer.save(bpe_dir, bpe_file)



