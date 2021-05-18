import random
import time
import numpy as np
import pandas as pd
import os
import argparse
import re
import multiprocessing as mp
from tqdm import tqdm
import pdb
# def define_argparser():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-ts', '--test_size', type=int, default=2000, help='Number of test data size')
#     args = parser.parse_args()
#     return args
if __name__ == '__main__':
    # args = define_argparser()
    # test_size = args.test_size
    # read excel files
    print('Reading excels ... ')
    excel_names = os.listdir('paired')
    excel_names = [f for f in excel_names if '.xlsx' in f]
    [print(f) for f in excel_names]
    print()
    dest_path = os.path.join('..', 'raw_koen')
    for i, file_name in enumerate(excel_names):
        if '조례' in file_name:
            domain = 'law'
        else:
            raise 'Invalid domain'
        print('Domain:', domain)
        file = pd.read_excel(os.path.join('paired', file_name))
        data_ko = file['원문'].values
        # test_ko = data_ko[:test_size]
        # dev_ko = data_ko[test_size:2*test_size]        
        # train_ko = data_ko[2*test_size:]
        lang = 'ko'
        # print(f'Split info ({lang}): {len(test_ko)} {len(dev_ko)} {len(train_ko)}')
        # with open(os.path.join(dest_path, f'{domain}-test.{lang}'), 'w', encoding='utf8') as f:
        #     for line in test_ko:
        #         f.write(str(line) + '\n')
        # with open(os.path.join(dest_path, f'{domain}-dev.{lang}'), 'w', encoding='utf8') as f:
        #     for line in dev_ko:
        #         f.write(str(line) + '\n')
        # with open(os.path.join(dest_path, f'{domain}-train.{lang}'), 'w', encoding='utf8') as f:
        #     for line in train_ko:
        #         f.write(str(line) + '\n')
        with open(os.path.join(dest_path, f'{domain}-all.{lang}'), 'w', encoding='utf8') as f:
            for line in data_ko:
                f.write(str(line) + '\n')
        data_en = file['번역문'].values
        # test_en = data_en[:test_size]
        # dev_en = data_en[test_size:2*test_size]        
        # train_en = data_en[2*test_size:]
        lang = 'en'
        # print(f'Split info ({lang}): {len(test_en)} {len(dev_en)} {len(train_en)}')
        # with open(os.path.join(dest_path, f'{domain}-test.{lang}'), 'w', encoding='utf8') as f:
        #     for line in test_en:
        #         f.write(str(line) + '\n')
        # with open(os.path.join(dest_path, f'{domain}-dev.{lang}'), 'w', encoding='utf8') as f:
        #     for line in dev_en:
        #         f.write(str(line) + '\n')
        # with open(os.path.join(dest_path, f'{domain}-train.{lang}'), 'w', encoding='utf8') as f:
        #     for line in train_en:
        #         f.write(str(line) + '\n')
        import pdb; pdb.set_trace()
        with open(os.path.join(dest_path, f'{domain}-all.{lang}'), 'w', encoding='utf8') as f:
            for line in data_en:
                f.write(str(line) + '\n')
        print(round((i+1)/len(excel_names)*100, 2), '% finished', sep='')
        print()
# Reading excels ... 
# 5_문어체_조례_200226.xlsx
# 4_문어체_한국문화_200226.xlsx
# Split info (ko): 2000 2000 96298
# Split info (en): 2000 2000 96298
# 50.0% finished
# Split info (ko): 2000 2000 96646
# Split info (en): 2000 2000 96646
# 100.0% finished