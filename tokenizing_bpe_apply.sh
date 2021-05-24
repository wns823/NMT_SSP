moses_scripts=./mosesdecoder/scripts # file path
bpe_scripts=./subword-nmt # file path

domain=$1
bpe_operations=20000 # 32000
S=de
T=en
data_dir=original_dataset
dest_dir=preprocess_dataset
split=$2 # valid, test

## 0. Remove unneccessary punctuation
perl $moses_scripts/tokenizer/normalize-punctuation.perl < $data_dir/${domain}-${split}.$S > $dest_dir/${domain}-${split}.$S
perl $moses_scripts/tokenizer/normalize-punctuation.perl < $data_dir/${domain}-${split}.$T > $dest_dir/${domain}-${split}.$T


### 1. Tokenizing ###
perl $moses_scripts/tokenizer/tokenizer.perl -threads 50 -l $S < $dest_dir/${domain}-${split}.$S > $dest_dir/${domain}-${split}.tok.$S
perl $moses_scripts/tokenizer/tokenizer.perl -threads 50 -l $T < $dest_dir/${domain}-${split}.$T > $dest_dir/${domain}-${split}.tok.$T


### 2. Truecaser ### 

## Apply ##
perl $moses_scripts/recaser/truecase.perl -model $dest_dir/${domain}-truecase-model.$S < $dest_dir/${domain}-${split}.tok.$S > $dest_dir/${domain}-${split}.tc.$S
perl $moses_scripts/recaser/truecase.perl -model $dest_dir/${domain}-truecase-model.$T < $dest_dir/${domain}-${split}.tok.$T > $dest_dir/${domain}-${split}.tc.$T

### 3. Apply bpe

## Apply ##
python3 $bpe_scripts/apply_bpe.py -c $dest_dir/$domain-${S}${T}.bpe < $dest_dir/${domain}-${split}.tc.$S > $dest_dir/${domain}-${split}.bpe.$S
python3 $bpe_scripts/apply_bpe.py -c $dest_dir/$domain-${S}${T}.bpe < $dest_dir/${domain}-${split}.tc.$T > $dest_dir/${domain}-${split}.bpe.$T



# bash tokenizing_bpe_apply.sh acquis valid