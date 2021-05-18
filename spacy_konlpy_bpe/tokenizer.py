import random
import time
import numpy as np
import pandas as pd
import os
import argparse
import re
import multiprocessing as mp

import spacy
spacy_en = spacy.load('en')
from konlpy.tag import Mecab
mecab = Mecab()


def post_tokenizer(raw_text, tokenized_text):
    '''
    나는 밥을 먹는다.
    =mecab=> 나 는 밥 을 먹 는다 .
    =post_tokenizer=> ▁나 는 ▁밥 을 ▁먹 는다 .

    =BPE=> ▁▁나 ▁는 ▁▁밥 ▁을 ▁▁먹 ▁는다 ▁.
    =undoBPE=> ▁▁나▁는▁▁밥▁을▁▁먹▁는다▁. (.replace(' ',''))
            => 나▁는 밥▁을 먹▁는다▁. (.replace('▁▁', ' '))
            => 나는 밥을 먹는다. (.replace('▁', ''))
    '''
    STR = '▁' # U+2581
    ref_tokens = raw_text.strip().split()

    idx = 0
    buf = []
    for ref_token in ref_tokens:
        tmp_buf = []
        while idx < len(tokenized_text):
            tmp_buf += [tokenized_text[idx]]
            idx+=1
            if ''.join(tmp_buf) == ref_token:
                break
        if len(tmp_buf) > 0:
            buf += [STR + tmp_buf[0].strip()] + tmp_buf[1:]
            
    return buf

# def data_process_ko(text):
#     text = re.sub(r'^[⓪①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳] ', '', text) # remove numbering
#     text = re.sub(r'^[0-9]+. ', '', text) # remove numbering
#     # text = re.sub(r'\([^가-힣]+\)', '', text) # remove (解囑)
#     return text
    
# def data_process_en(text):
#     text = re.sub(r'^\([0-9]\) ', '', text) # remove numbering
#     text = re.sub(r'^[0-9]+. ', '', text) # remove numbering
#     text = text.replace('“', '"').replace('”', '"')
#     if text[-1]==':' or text[-1]==';':
#         text = text[:-1]
#     return text

def tokenize_ko(raw_text):
    # processed_text = data_process_ko(raw_text)
    tokenized_text = [tok[0] for tok in mecab.pos(raw_text)]
    return post_tokenizer(raw_text, tokenized_text)

def tokenize_en(raw_text):
    # processed_text = data_process_en(raw_text)
    tokenized_text = [tok.text for tok in spacy_en.tokenizer(raw_text)]
    return post_tokenizer(raw_text, tokenized_text)

def define_argparser():

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', type=str, required=True, help='File path')
    parser.add_argument('-o', '--output', type=str, required=True, help='Output path')
    parser.add_argument('--threads', type=int, default=-1, help='Num threads')
    # parser.add_argument('-mc', '--monolingual_corpus', action="store_true", help='Preprocess monolingual data')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = define_argparser()

    input_path = args.filename
    output_path = args.output
    # isMonolingual = args.monolingual_corpus

    if args.threads == -1:
        num_process = mp.cpu_count()
    else:
        num_process = args.threads
        

    # Tokenize
    if input_path.endswith('.ko'):
        func = tokenize_ko
    elif input_path.endswith('.en'):
        func = tokenize_en
    else:
        raise 'Invalid language pairs'

    
    with open(input_path, 'r', encoding='utf8') as f:
        raw_data = f.read().split('\n')
    raw_data = [line for line in raw_data if len(line)>0]

    pool = mp.Pool(processes=num_process)
    tokenized_data = pool.map(func, raw_data)
    pool.close()
    pool.join()

    tokenized_data = [' '.join(line) for line in tokenized_data]

    with open(output_path, 'w', encoding='utf8') as f:
        for line in tokenized_data:
            f.write(str(line) + '\n')



    # if isMonolingual:    

    #     if domain=='law':
    #         raw_data_dir = os.path.join('raw_data_mono', 'domain_specific', 'law')
    #     elif domain=='culture':
    #         raw_data_dir = os.path.join('raw_data_mono', 'domain_specific', 'culture')        

    #     if domain in ['law', 'culture']:

    #         mono_excel_names = os.listdir(raw_data_dir)
    #         mono_excel_names = [f for f in mono_excel_names if '.xlsx' in f]
    #         [print(f) for f in mono_excel_names + excel_names]

    #         data_ko = []
    #         data_en = []
    #         for i, file_name in enumerate(mono_excel_names):
    #             file = pd.read_excel(os.path.join(raw_data_dir, file_name))
    #             data_ko.extend(file['KOR'].values)
    #             data_en.extend(file['ENG'].values)
    #             print(round((i+1)/len(mono_excel_names)*100, 2), '% finished', sep='')


    #         # Concat dataset
    #         data_ko_preprocessed = raw_data_ko 
    #         data_en_preprocessed = raw_data_en  

    #         pool = mp.Pool(processes=mp.cpu_count())
    #         result_ko = pool.map(tokenize_ko, data_ko_preprocessed)
    #         result_en = pool.map(tokenize_en, data_en_preprocessed)
    #         pool.close()
    #         pool.join()

    #         data_ko_lengths = []
    #         for text_ko in result_ko:
    #             data_ko_lengths.append(len(text_ko))

    #         data_en_lengths = []   
    #         for text_en in result_en:
    #             data_en_lengths.append(len(text_en))
                
    #         # Calculating max seq length
    #         print(f'Filtering out {max_seq_len_percentile}th percentile or above')
    #         max_len_ko = np.percentile(data_ko_lengths, q=max_seq_len_percentile)
    #         max_len_en = np.percentile(data_en_lengths, q=max_seq_len_percentile)


    #         # Korean
    #         data_ko_ = []
    #         for text_ko in tqdm(result_ko):
    #             if len(text_ko) > 3 and len(text_ko) <= max_len_ko:
    #                 data_ko_.append(' '.join(text_ko))
    #         print(f'Maximum length (Korean): {max_len_ko}')
    #         train_ko, test_ko = train_test_split(data_ko_+train['ko'].values.tolist(), test_size=test_size*2, random_state=1)
    #         valid_ko, test_ko = train_test_split(test_ko, test_size=test_size, random_state=1)
    #         print(len(data_ko_lengths), '=>', len(train_ko)+len(valid_ko)+len(test_ko))
    #         print(len(train_ko), '+', len(valid_ko), '+', len(test_ko), '=', len(train_ko)+len(valid_ko)+len(test_ko)) 


    #         # English
    #         data_en_ = []
    #         for text_en in tqdm(result_en):
    #             if len(text_en) > 3 and len(text_en) <= max_len_en:
    #                 data_en_.append(' '.join(text_en))
    #         print(f'Maximum length (English): {max_len_en}')
    #         train_en, test_en = train_test_split(data_en_+train['en'].values.tolist(), test_size=test_size*2, random_state=2)
    #         valid_en, test_en = train_test_split(test_en, test_size=test_size, random_state=2)
    #         print()
    #         print(f'Saving monolingual data...')
    #         print(len(data_en_lengths), '=>', len(train_en)+len(valid_en)+len(test_en))
    #         print(len(train_en), '+', len(valid_en), '+', len(test_en), '=', len(train_en)+len(valid_en)+len(test_en))         

    #         if domain=='law':
    #             new_data_dir = os.path.join('corpus_mono', 'domain_specific', 'law')
    #         elif domain=='culture':
    #             new_data_dir = os.path.join('corpus_mono', 'domain_specific', 'culture')  
    #         if not os.path.exists(new_data_dir):
    #             os.makedirs(new_data_dir)

    #         save_file(data=train_ko, path=new_data_dir, filename="train_corpus.ko")
    #         save_file(data=valid_ko, path=new_data_dir, filename="val_corpus.ko")
    #         save_file(data=test_ko, path=new_data_dir, filename="test_corpus.ko")
    #         save_file(data=train_en, path=new_data_dir, filename="train_corpus.en")
    #         save_file(data=valid_en, path=new_data_dir, filename="val_corpus.en")
    #         save_file(data=test_en, path=new_data_dir, filename="test_corpus.en")
    #         print()
    
