src=$1
lng=$2
domain=$3
echo "tgt lng $lng"

if [ $src = "ko" ]
then
    for sub in train valid test
    do
        python undo_bpe_ko.py --i ${domain}_${src}${lng}/${sub}.${lng} > ${domain}_${src}${lng}/${sub}.tok.${lng}
    done
else
    for sub in train valid test
    do
        sed -r 's/(@@ )|(@@ ?$)//g' ${domain}_${src}${lng}/${sub}.${lng} > ${domain}_${src}${lng}/${sub}.tok.${lng}
        # mosesdecoder/scripts/tokenizer/detokenizer.perl -l $lng < ${domain}_dataset/${sub}.bert.${lng}.tok > ${domain}_dataset/${sub}.bert.${lng}
        # rm ${domain}_dataset/${sub}.bert.${lng}.tok
    done
fi