import argparse, os


# 전체 unique term , 맞춰야할 term, exact matching, partial matching

def make_sub(phrase) :
    res = []
    sub_list = phrase.split(" ")
    n = len(phrase.split(" "))
    for step in range(1, n)  :
        for i in range(0, n - step) :
            #print(i , i + step)
            res.append( " ".join(sub_list[i:i + step + 1]) )  

    return res


def get_stat(sentence, phrase):
    if phrase in sentence :
        return len(phrase.split(" ")), phrase
    else :
        compare_list = make_sub(phrase)
        for compare in reversed(compare_list) :
            if compare in sentence :
                return len(compare.split(" ")), compare
        return 0, 0

if __name__ == "__main__" :

    parser = argparse.ArgumentParser()

    # python ngram_inference.py --domain acquis --src_lang de --tgt_lang en --outputfile result_collection/law_leca_span_with_dict.txt
    # python ngram_inference.py --domain emea --src_lang de --tgt_lang en --outputfile rebutal_emea_leca_span.txt
    # python ngram_inference.py --domain law --src_lang ko --tgt_lang en --outputfile law_leca_span_with_dict.txt

    parser.add_argument('--domain',  type=str, default="acquis", help='Type the domain' )
    parser.add_argument('--src_lang',  type=str, default="de", help='Type the source language' )
    parser.add_argument('--tgt_lang',  type=str, default="en", help='Type the target language' )
    parser.add_argument('--outputfile',  type=str, default="", help='Type the output file' )
    
    args = parser.parse_args()
    
    domain = args.domain
    src_lang = args.src_lang
    tgt_lang = args.tgt_lang
    output_file = args.outputfile

    pred = open( f'{output_file}', 'r').readlines()
    
    phrase = open( f'phrase/{domain}-phrase-test.{src_lang}-{tgt_lang}', 'r').readlines() 

    
    #################################################################################

    matched_phrase = {}
    partial_phrase = {} # 

    reference_phrase = {} # 분모

    for i in range(len(phrase)) :

        phrase_ = phrase[i].replace("\n", "")

        if phrase_ != "梁" :
            phrase_list = []
            for e in phrase_.split(" ||| ") :
                phrase_list.append( e.split(":")[1] )

            for p in phrase_list :
                original_n = len(p.split(" "))

                if original_n in reference_phrase.keys() :
                    reference_phrase[original_n].append(p)
                else :
                    reference_phrase[original_n] = [p]

                n , phrase_after = get_stat(pred[i], p)
                if n == original_n :
                    if n in matched_phrase.keys() :
                        matched_phrase[n].append(phrase_after)
                    else :
                        matched_phrase[n] = [phrase_after]
                elif n != 0 :
                    if original_n in partial_phrase.keys() :
                        partial_phrase[original_n].append(phrase_after)
                    else :
                        partial_phrase[original_n] = [phrase_after]
    
    # ngram_list = reference_phrase.keys() 
    ngram_list = sorted(reference_phrase.keys(), reverse=False)

    
    tur_list = []
    iou_list = []
    reference_list = []

    for k in ngram_list :
        partial_score = 0.0
        match_score = 0.0

        if k in matched_phrase.keys() :
            match_score += len(matched_phrase[k])

        if k in partial_phrase.keys() :
            for e in partial_phrase[k]:
                sub = len(e.split(" "))
                partial_score = partial_score + sub / k 

        TUR =  match_score  / len(reference_phrase[k])
        
        if k >= 3 :
            tur_list.append(TUR)
    
        print( "TUR N-gram %d : %.4f" % (k, TUR ) )

        if k >= 3 :
            reference_list.append(len(reference_phrase[k]))

        IOU = ( match_score + partial_score ) / len(reference_phrase[k])
    
        if k >= 3 :
            iou_list.append(IOU)
    
        print( "IOU N-gram %d : %.4f" % (k, IOU ) )
        print()
    
    micro_tur = 0.0
    micro_iou = 0.0

    for t, i, r in zip(tur_list, iou_list, reference_list) :
        micro_tur += t * r
        micro_iou += i * r

    # print("reference_list")
    # print(reference_list)

    print("Micro TUR : %.4f" % (micro_tur / sum(reference_list)) )
    print("Micro IOU : %.4f" % (micro_iou / sum(reference_list)) )

    # # print macro
    print("Macro TUR : %.4f" % ( sum(tur_list) / len(tur_list) ) )
    print("Macro IOU : %.4f" % ( sum(iou_list) / len(iou_list) ) )
