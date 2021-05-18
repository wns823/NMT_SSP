export CUDA_VISIBLE_DEVICES=$1

domain=$2 # acquis, emea
sl=$3
tl=$4
model_path=$5
testclean=$6

data_dir=${PWD}/data-bin/${domain}_${sl}${tl}

if [[ $sl == "ko" ]]; then
    preprocess="sentencepiece"
else
    preprocess="@@ "
fi


if [[ $testclean == "1" ]]; then
    CUDA_VISIBLE_DEVICES=1 fairseq-generate ${data_dir} -s ${3} -t ${4} \
                --path outputs/${model_path}/checkpoint_best.pt \
                --batch-size 64 --remove-bpe ${preprocess} \
                --consnmt --task translation_leca --use-ptrnet --testclean \
                --model-overrides "{'beam':5}"
else
    CUDA_VISIBLE_DEVICES=1 python generate_text.py ${data_dir} -s ${3} -t ${4} \
                --path outputs/${model_path}/checkpoint_best.pt \
                --batch-size 64 --remove-bpe ${preprocess} \
                --consnmt --task translation_leca --use-ptrnet \
                --model-overrides "{'beam':5}"
fi


# bash inference.sh 0 acquis de en model_path 0