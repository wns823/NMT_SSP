import os
import argparse
import json
import tqdm

# first run 'make_tok.sh' then run 'mask_span.py'

def subspan_generator(trg_bpe_line, trg_tok_line, lang) :
    if lang == 'ko' :
        bpe = '▁'
    else :
        bpe = '@@'
    trg_bpe_line_split = trg_bpe_line.split()
    trg_tok_line_split = trg_tok_line.split()
    idx = 0
    spans = []
    for ref_token in trg_tok_line_split:
        tmp_buf = []
        tmp_idx = 0
        i = idx
        while idx < len(trg_bpe_line_split):
            tmp_buf += [trg_bpe_line_split[idx].replace(bpe,'')]
            idx+=1
            tmp_idx+=1
            if ''.join(tmp_buf) == ref_token.replace(bpe,''):
                break
        if len(tmp_buf) > 0:
            spans += [[i, i+tmp_idx]]
    return spans

if __name__ == "__main__" :
    
    parser = argparse.ArgumentParser()

    # python make_span.py --directory law_dataset --src ko --tgt en --saved data-bin/law_koen
    # load model
    parser.add_argument('--directory', required=True, help='File path')

    parser.add_argument('--src', type=str, default='ko', help='Source language')
    parser.add_argument('--tgt', type=str, default='en', help='Source language')
    parser.add_argument('--saved', type=str, default='', help='Source language')


    args = parser.parse_args()

    split = ["train", "valid", "test"]
    src = args.src
    tgt = args.tgt
    directory = args.directory
    saved = args.saved

    for s in split :
        tgt_bpe_line = open(f"{directory}/{s}.{tgt}", "r").readlines()
        tgt_tok_line = open(f"{directory}/{s}.tok.{tgt}", "r").readlines()

        span = {}        

        for i, (bpe, tok) in enumerate( zip(tgt_bpe_line, tgt_tok_line) ) :
            sample = subspan_generator(bpe, tok, src)
            span[int(i)] = {}
            anchor_list = [i for i in range( len(bpe.split(" ")) )]
            
            for a in anchor_list :
                for sam in sample :
                    if sam[0] <= a and a < sam[1] :
                        span[int(i)][int(a)] = sam
             
            
        with open(f"{saved}/{s}_span.json", "w") as f :
            json.dump( span, f)
        
            


