export CUDA_VISIBLE_DEVICES=$1 
domain=$2 # acquis, emea
sl=$3 # ko
tl=$4 # en
project=$5
scheme=$6
ratio=$7
lower=$8
upper=$9
dropout=${10}

data_dir=${PWD}/data-bin/${domain}_${sl}${tl}/

save_dir=${PWD}/outputs/${project}/
mkdir -p $save_dir


fairseq-train $data_dir \
            --arch transformer_leca \
            --task translation_leca \
            --criterion label_smoothed_cross_entropy_cbert \
            --scheme $scheme --span-lower $lower --span-upper $upper --geometric-p 0.2 --mask-ratio $ratio \
            --share-all-embeddings \
            --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
            --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
            --dropout ${dropout}  \
            --label-smoothing 0.1 --weight-decay 0.0 \
            --max-tokens 4096 --save-dir $save_dir \
            --update-freq 2 --no-progress-bar --log-format json --log-interval 50 \
            --eval-bleu --patience 100 \
            --eval-bleu-args '{"beam": 5}' \
            --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
            --save-interval-updates  500 --keep-interval-updates 20 \
            --consnmt --use-ptrnet --test-seed 1 1> $save_dir/log 2> $save_dir/err
 
# bash train.sh 3 acquis de en unique_acquis_leca_span_0.3 span 0.5 1 10 0.3
# bash train.sh 2 emea de en unique_emea_leca_span_0.3 span 0.5 1 10 0.3
