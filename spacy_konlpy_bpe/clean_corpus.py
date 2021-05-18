import os
import argparse

from tqdm import tqdm
from tokenizers import SentencePieceBPETokenizer
import multiprocessing as mp



def define_argparser():

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--inputs', type=str, required=True, nargs=2, help='File paths')
    parser.add_argument('-o', '--outputs', type=str, required=True, help='Input file path')

    parser.add_argument('-pp', '--parallel_train_percent', type=float, default=1.0, required=True, help='Percentage of parallel train data')
    # parser.add_argument('--src_monolingual_train_percent', type=float, default=1.0,required=True,  help='Percentage of source monolingual train data')
    # parser.add_argument('--trg_monolingual_train_percent', type=float, default=1.0, required=True, help='Percentage of target monolingual train data')    

    parser.add_argument('--max_src', type=int, default=50, help='Maximum source sequence')
    parser.add_argument('--max_trg', type=int, default=50, help='Maximum target sequence')

    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = define_argparser()

    input_files = args.inputs
    output_files = args.outputs
    max_src = args.max_src
    max_trg = args.max_trg

    parallel_train_percent = args.parallel_train_percent
    # src_monolingual_train_percent = args.src_monolingual_train_percent
    # trg_monolingual_train_percent = args.trg_monolingual_train_percent


    with open(input_files[0], 'r', encoding='utf8') as f:
        corpus_src = f.read().split('\n')
    with open(input_files[1], 'r', encoding='utf8') as f:
        corpus_trg = f.read().split('\n')

    
    filtered_corpus_src = []
    filtered_corpus_trg = []
    for line_src, line_trg in zip(corpus_src, corpus_trg):
        line_src = line_src.split()
        line_trg = line_trg.split()
        if len(line_src) <= max_src and len(line_trg) <= max_trg and len(line_src) > 0 and len(line_trg) > 0:
            filtered_corpus_src.append(' '.join(line_src))
            filtered_corpus_trg.append(' '.join(line_trg))


    train_partial_src = filtered_corpus_src[:int(len(filtered_corpus_src)*parallel_train_percent)]
    # if args.monolingual_corpus:
    #     train_corpus_mono_ko_portion = train_corpus_mono_ko[:int(len(train_corpus_mono_ko)*src_monolingual_train_percent)]

    train_partial_trg = filtered_corpus_trg[:int(len(filtered_corpus_trg)*parallel_train_percent)]
    # if args.monolingual_corpus:
    #     train_corpus_mono_ko_portion = train_corpus_mono_ko[:int(len(train_corpus_mono_ko)*src_monolingual_train_percent)]

    assert len(train_partial_src)==len(train_partial_trg)



    ## infer langauge
    src_lang = input_files[0].split('.')[-1]
    trg_lang = input_files[1].split('.')[-1]


    if parallel_train_percent != 1.0:
        output_files = output_files+str(parallel_train_percent)

    with open(f'{output_files}.{src_lang}', 'w', encoding='utf8') as f:
        for line in tqdm(train_partial_src):
            f.write(str(line) + '\n')

    with open(f'{output_files}.{trg_lang}', 'w', encoding='utf8') as f:
        for line in tqdm(train_partial_trg):
            f.write(str(line) + '\n')


# 100%|████████████████████████████| 96298/96298 [00:00<00:00, 375452.43it/s]
# 100%|████████████████████████████| 96298/96298 [00:00<00:00, 462532.29it/s]
# 100%|████████████████████████████| 69837/69837 [00:00<00:00, 106742.76it/s]
# 100%|████████████████████████████| 69837/69837 [00:00<00:00, 149752.00it/s