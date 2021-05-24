domain=$1 # law
source_lang=$2 # ko
target_lang=$3 # en
split=$4 # dev, tests
pp=1.0
# max_src=60
# max_trg=60
DIRECTORY=./raw_${source_lang}${target_lang}
if [ -d "$DIRECTORY" ]; then
    data_dir=$DIRECTORY
else
    data_dir=./raw_${target_lang}${source_lang}
fi
bpe_dir=./dataset_${source_lang}${target_lang}
dest_dir=./dataset_${source_lang}${target_lang}
mkdir -p $dest_dir
type=$domain-$split # dev, test
S=$source_lang
T=$target_lang
bpe_operations=20000 # 32000
# if [ $source_lang == "ko" || $target_lang == "ko" ]; then
preprocess_scripts=./spacy_konlpy_bpe # file path
### 1. Tokenizing ###
python3 $preprocess_scripts/tokenizer.py --filename $data_dir/$type.$S --output $dest_dir/$type.tok.$S
python3 $preprocess_scripts/tokenizer.py --filename $data_dir/$type.$T --output $dest_dir/$type.tok.$T
### 2. Apply BPE ###
python3 $preprocess_scripts/apply_bpe.py -c $bpe_dir/$domain-${S}${T}${bpe_operations} -i $dest_dir/$type.tok.$S -o $dest_dir/$type.bpe.$S
python3 $preprocess_scripts/apply_bpe.py -c $bpe_dir/$domain-${S}${T}${bpe_operations} -i $dest_dir/$type.tok.$T -o $dest_dir/$type.bpe.$T