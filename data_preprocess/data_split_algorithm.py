import json, os, re, parmap
import multiprocessing as mp
from multiprocessing import Manager
import numpy as np
import codecs
import argparse

def matching(index, tgt_corpus, src_corpus, ngram_list, ngram, dictionary, unique, tag) :
    
    for i in index :
        sub1 = []
        sub2 = []
        rep_sentence = tgt_corpus[i]
        for n in ngram_list :
            for sub_n in ngram[n] :
                if sub_n in rep_sentence:
                    for s in dictionary[sub_n] : 
                        try : 
                            if type(s) != float and s in src_corpus[i] :
                    
                                rep_sentence = rep_sentence.replace(sub_n, "", 1) # word delete
                                
                                sub1.append( [n, s, sub_n])
                                sub2.append(n)
                                break
                        except :
                            print(sub_n)
                            import pdb; pdb.set_trace()
        unique[i] = sub1
        tag[i] = sub2

def overlap_filter(index, n_gram_str, n_gram_index, tag_, tgt_corpus) :

    sub_str = []
    sub_index = []

    for i in tag_[index] :
        if tgt_corpus[i] in sub_str :
            n_index = sub_str.index(tgt_corpus[i])
            sub_index[n_index].append(i)
        else :
            sub_str.append(tgt_corpus[i])
            sub_index.append( [i] )

    n_gram_str[index] = sub_str
    n_gram_index[index] = sub_index


if __name__ == "__main__" :
    parser = argparse.ArgumentParser()

    # python split_data.py --domain emea --src_path raw_data/EMEA.de-en.de  --tgt_path raw_data/EMEA.de-en.en --directory_path dictionary/iate_en_de_filter.json --src_lang de
    # python split_data.py --domain acquis --src_path raw_data/JRC-Acquis.de-en.de  --tgt_path raw_data/JRC-Acquis.de-en.en --directory_path dictionary/iate_en_de_filter.json --src_lang de
    # python split_data.py --domain law --src_path raw_data/law-all.ko  --tgt_path raw_data/law-all.en --directory_path dictionary/dict_law_en_ko.json --src_lang ko

    parser.add_argument('--domain',  type=str, default="acquis", help='Type the domain' )
    parser.add_argument('--src_path',  type=str, default="acquis", help='Type the source data' )
    parser.add_argument('--tgt_path',  type=str, default="acquis", help='Type the target data' )
    parser.add_argument('--directory_path',  type=str, default="dictionary/iate_en_de_filter.json", help='dictionary/dict_law_en_ko.json, dictionary/iate_en_de_filter.json' )
    parser.add_argument('--src_lang',  type=str, default="de", help='Type the source language' )
    parser.add_argument('--tgt_lang',  type=str, default="en", help='Type the target language' )
    parser.add_argument('--test_size',  type=str, default="3000", help='Type the test size' )
    


    args = parser.parse_args()
    
    dictionary_path = args.directory_path
    domain = args.domain
    src_lang = args.src_lang
    tgt_lang = args.tgt_lang
    
    test_size = int(args.test_size)

    saved_directory = "original_datset"

    ####################################################################################################################

    src_path = args.src_path
    tgt_path = args.tgt_path
    
    with open( dictionary_path, "r") as f:
        dictionary = json.load(f)

    src = open(src_path, "r")
    tgt = open(tgt_path, "r")

    src_corpus = src.readlines()
    tgt_corpus = tgt.readlines()

    data_size = len(tgt_corpus)

    num_cores = mp.cpu_count()     

    index = list(range( 0, data_size ))
    splited_index =  np.array_split(index, num_cores)
    splited_index = [x.tolist() for x in splited_index]


    manager = Manager()

    tgt_dict_list = dictionary.keys() # english
    ngram = {}

    for s in tgt_dict_list :
        sub_word = s.split(" ")
        num = len(sub_word)

        if num not in ngram.keys() :
            ngram[ num ] = [s]
        else :
            ngram[ num ].append(s)                    

    ngram_list = sorted(ngram.keys(), reverse=True)


    unique = manager.dict()
    tag = manager.dict()

    tag_ = {}

    parmap.map(matching, splited_index, tgt_corpus, src_corpus, ngram_list, ngram, dictionary, unique, tag, pm_pbar=True, pm_processes=num_cores)

    dummy_data = []
    # tag -> tag_ (i -> n)
    for i in tag.keys() :
        if len(tag[i]) != 0:
            if max(tag[i]) in tag_.keys() :
                tag_[ max(tag[i]) ].append(i)
            else :
                tag_[ max(tag[i]) ] = [i]
        else :
            dummy_data.append(i)


    ngram_list = tag_.keys() 
    ngram_list = sorted(tag_.keys(), reverse=True)
    
    ####################################################################################################################
    # In this section, data split is executed by duplicate sentences and unique sentences. we define this process DuplicateSampling(), UniqueSampling() in our paper.
    
    # test_size = 3000
    ratio = test_size / len(src_corpus)

    n_gram_str = manager.dict()
    n_gram_index = manager.dict()

    parmap.map(overlap_filter, ngram_list, n_gram_str, n_gram_index, tag_, tgt_corpus, pm_pbar=True, pm_processes=len(ngram_list))

    n_k = n_gram_index.keys()

    train = []
    valid = []
    test = []

    for n in n_k :
        sub_list = []
        for element in n_gram_index[n] :
            if len(element) >= 2 :
                if ratio *len(element) >= 1 :
                    split = int(ratio *len(element))
                    test += element[:split]
                    valid += element[split:2*split]
                    train += element[2*split:]
                else :
                    valid.append(element[0])
                    test.append(element[1])
                    if len(element) != 2:
                        train += element[2:]
            else :
                sub_list += element
        
        if len(sub_list) >= 3 :
            train += sub_list[::3]
            valid += sub_list[1::3]
            test += sub_list[2::3]
        elif len(sub_list) == 2 :
            train.append(sub_list[0])
            valid.append(sub_list[1])
        elif len(sub_list) == 1 : 
            test.append(sub_list[0])
            

    if len(test) > test_size :
        train += test[test_size:]
        test = test[:test_size]

    if len(test) < test_size :
        residual = test_size - len(test)
        pin = len(train)
        test += train[: pin - 1 :-1]
        train = train[: pin]

    if len(valid) > test_size :
        train += valid[test_size:]
        valid = valid[:test_size]

    if len(valid) < test_size :
        residual = test_size - len(valid)
        pin = len(train)
        valid += train[: pin - 1 :-1]
        train = train[: pin]


    ####################################################################################################################


    t_1 = open(f"{saved_directory}/{domain}-train.{src_lang}", "w")
    t_2 = open(f"{saved_directory}/{domain}-train.{tgt_lang}", "w")
    t_3 = open(f"{saved_directory}/{domain}-phrase-train.{src_lang}-{tgt_lang}", "w")

    for i in train :
        t_1.write(src_corpus[i]) 
        t_2.write(tgt_corpus[i])
        if len(unique[i]) != 0:
            t_3.write(" ||| ".join([ ":".join([ e[1], e[2] ]) for e in unique[i]]) + '\n' )
        else : ############### 
            t_3.write("梁\n") # If sentence is not matched with dictionary, it writes dummy value for next preprocessing's convenience.

    t_1.close()
    t_2.close()
    t_3.close()       

    v_1 = open(f"{saved_directory}/{domain}-valid.{src_lang}", "w")
    v_2 = open(f"{saved_directory}/{domain}-valid.{tgt_lang}", "w")
    v_3 = open(f"{saved_directory}/{domain}-phrase-valid.{src_lang}-{tgt_lang}", "w")

    for i in valid :
        v_1.write(src_corpus[i])
        v_2.write(tgt_corpus[i])
        if len(unique[i]) != 0:
            v_3.write(" ||| ".join([ ":".join([ e[1], e[2] ]) for e in unique[i]]) + '\n' )
        else :
            v_3.write("梁\n") # If sentence is not matched with dictionary, it writes dummy value for next preprocessing's convenience.

    v_1.close()
    v_2.close()       
    v_3.close()

    tt_1 = open(f"{saved_directory}/{domain}-test.{src_lang}", "w")
    tt_2 = open(f"{saved_directory}/{domain}-test.{tgt_lang}", "w")
    tt_3 = open(f"{saved_directory}/{domain}-phrase-test.{src_lang}-{tgt_lang}", "w")

    for i in test :
        tt_1.write(src_corpus[i])
        tt_2.write(tgt_corpus[i])
        if len(unique[i]) != 0:
            tt_3.write(" ||| ".join([ ":".join([ e[1], e[2] ]) for e in unique[i]]) + '\n' )
        else :
            tt_3.write("梁\n") # If sentence is not matched with dictionary, it writes dummy value for next preprocessing's convenience.


    tt_1.close()
    tt_2.close()
    tt_3.close()       

