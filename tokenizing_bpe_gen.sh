moses_scripts=./mosesdecoder/scripts # file path
bpe_scripts=./subword-nmt # file path

domain=$1
bpe_operations=20000 # 32000
S=de
T=en
data_dir=original_datset/${domain}_${S}${T}
dest_dir=preprocess_dataset/${domain}_${S}${T}
split=train # law-dev, law-test

## 0. Remove unneccessary punctuation
perl $moses_scripts/tokenizer/normalize-punctuation.perl < $data_dir/${split}.$S > $dest_dir/${domain}-${split}.$S
perl $moses_scripts/tokenizer/normalize-punctuation.perl < $data_dir/${split}.$T > $dest_dir/${domain}-${split}.$T


## 1. Tokenizing
perl $moses_scripts/tokenizer/tokenizer.perl -threads 50 -l $S < $dest_dir/${domain}-${split}.$S > $dest_dir/${domain}-${split}.tok.$S
perl $moses_scripts/tokenizer/tokenizer.perl -threads 50 -l $T < $dest_dir/${domain}-${split}.$T > $dest_dir/${domain}-${split}.tok.$T


## 2. Truecaser 
## Train
perl $moses_scripts/recaser/train-truecaser.perl -corpus $dest_dir/${domain}-${split}.tok.$S -model $dest_dir/$domain-truecase-model.$S
perl $moses_scripts/recaser/train-truecaser.perl -corpus $dest_dir/${domain}-${split}.tok.$T -model $dest_dir/$domain-truecase-model.$T

## Apply
perl $moses_scripts/recaser/truecase.perl -model $dest_dir/${domain}-truecase-model.$S < $dest_dir/${domain}-${split}.tok.$S > $dest_dir/${domain}-${split}.tc.$S
perl $moses_scripts/recaser/truecase.perl -model $dest_dir/${domain}-truecase-model.$T < $dest_dir/${domain}-${split}.tok.$T > $dest_dir/${domain}-${split}.tc.$T

## 3. apply bpe
## Train
python3 $bpe_scripts/learn_joint_bpe_and_vocab.py -i $dest_dir/${domain}-${split}.tc.$S $dest_dir/${domain}-${split}.tc.$T --write-vocabulary $dest_dir/vocab.$S $dest_dir/vocab.$T -s $bpe_operations -o $dest_dir/$domain-${S}${T}.bpe 

## Apply
python3 $bpe_scripts/apply_bpe.py -c $dest_dir/$domain-${S}${T}.bpe < $dest_dir/${domain}-${split}.tc.$S > $dest_dir/${domain}-${split}.bpe.$S
python3 $bpe_scripts/apply_bpe.py -c $dest_dir/$domain-${S}${T}.bpe < $dest_dir/${domain}-${split}.tc.$T > $dest_dir/${domain}-${split}.bpe.$T
