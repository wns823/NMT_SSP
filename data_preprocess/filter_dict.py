import os
import json 
import codecs

if __name__ == "__main__" :

    dictionary = json.load(codecs.open('dictionary/iate_en_de_all.json', 'r', 'utf-8-sig'))
    eng = dictionary.keys()

    new_dict = {}

    for k in eng :
        n = k.split(" ")
        if len(k) >= 4 and len(n) <= 20 :
            new_dict[k] = dictionary[k]

    with open('dictionary/iate_en_de_filter.json', 'w' , encoding='utf-8') as f:
        json.dump( new_dict, f, ensure_ascii=False)
