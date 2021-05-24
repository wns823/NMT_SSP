import os
import argparse

# python filter_data.py --domain emea --src_lang de --tgt_lang en --split train --min 5 --max 80

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()

    parser.add_argument('--domain',  type=str, default="acquis", help='Type the domain' )
    parser.add_argument('--src_lang',  type=str, default="de", help='Type the source language' )
    parser.add_argument('--tgt_lang',  type=str, default="en", help='Type the target language' )
    parser.add_argument('--split',  type=str, default="train", help='Type the split' )
    parser.add_argument('--min',  type=str, default="train", help='Type the min' )
    parser.add_argument('--max',  type=str, default="train", help='Type the max' )

    args = parser.parse_args()

    domain = args.domain
    src_lang = args.src_lang
    tgt_lang = args.tgt_lang
    split = args.split # train, dev, test

    
    src = open(f"preprocess_dataset/{domain}-{split}.bpe.{src_lang}", "r") # bpe.clean.
    tgt = open(f"preprocess_dataset/{domain}-{split}.bpe.{tgt_lang}", "r") # bpe.clean.
    phrase = open(f"original_dataset/{domain}-phrase-{split}.{src_lang}-{tgt_lang}", "r")

    src2 = open(f"{domain}_{src_lang}{tgt_lang}/{split}.{src_lang}", "w") # bpe.clean.
    tgt2 = open(f"{domain}_{src_lang}{tgt_lang}/{split}.{tgt_lang}", "w") # bpe.clean.
    phrase2 = open(f"{domain}_{src_lang}{tgt_lang}/{domain}-phrase-{split}.{src_lang}-{tgt_lang}", "w")


    src_corpus = src.readlines()
    tgt_corpus = tgt.readlines()
    phrase_corpus = phrase.readlines()

    min_ = int(args.min)
    max_ = int(args.max)

    src_ = []
    tgt_ = []
    phrase_ = []

    for i, s in enumerate(src_corpus) :
        length = len(s.split(" "))
        if min_ <= length and length <= max_ :
            length = len(tgt_corpus[i].split(" "))
            if min_ <= length and length <= max_ :
                src2.write(s)
                tgt2.write(tgt_corpus[i])
                phrase2.write(phrase_corpus[i])
    src2.close()
    tgt2.close()
    phrase2.close()

    src.close()
    tgt.close()
    phrase.close()