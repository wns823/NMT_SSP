'''
It is applied for IATE dictionary. In KO-EN dictionary, It was preprocessed by hand, and since all words are related to the law, there is no need to filter.
'''

import os
import json 
import codecs
import argparse

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument('--min',  type=str, default="0", help='Type the min n-gram word' )
    parser.add_argument('--max',  type=str, default="100", help='Type the max n-gram word' )

    args = parser.parse_args()
    
    ngram_min = int(args.min)
    ngram_max = int(args.max)

    dictionary = json.load(codecs.open('dictionary/iate_en_de_all.json', 'r', 'utf-8-sig'))
    eng = dictionary.keys()

    new_dict = {}

    for k in eng :
        n = k.split(" ")
        if len(k) >= ngram_min and len(n) <= ngram_max :
            new_dict[k] = dictionary[k]

    with open('dictionary/iate_en_de_filter.json', 'w' , encoding='utf-8') as f:
        json.dump( new_dict, f, ensure_ascii=False)
