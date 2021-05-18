import os

if __name__ == "__main__" :

    domain = "emea"
    src_lang = "de"
    tgt_lang = "en"
    split = "test" # train, dev, test

    src = open(f"dataset_{src_lang}{tgt_lang}/{domain}-{split}.bpe.{src_lang}", "r") # bpe.clean.
    tgt = open(f"dataset_{src_lang}{tgt_lang}/{domain}-{split}.bpe.{tgt_lang}", "r") # bpe.clean.
    phrase = open(f"dataset_{src_lang}{tgt_lang}/{domain}-phrase-{split}.{src_lang}-{tgt_lang}", "r")


    src2 = open(f"{domain}_{src_lang}{tgt_lang}/{domain}-{split}.bpe.{src_lang}", "w") # bpe.clean.
    tgt2 = open(f"{domain}_{src_lang}{tgt_lang}/{domain}-{split}.bpe.{tgt_lang}", "w") # bpe.clean.
    phrase2 = open(f"{domain}_{src_lang}{tgt_lang}/{domain}-phrase-{split}.{src_lang}-{tgt_lang}", "w")

    src_corpus = src.readlines()
    tgt_corpus = tgt.readlines()
    phrase_corpus = phrase.readlines()

    min_ = 5
    max_ = 80

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