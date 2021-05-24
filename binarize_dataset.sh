domain=$1 # law, acquis, emea
sl=$2 # de
tl=$3 # en

data_dir=${domain}_${sl}${tl}
out_dir=data-bin

fairseq-preprocess --source-lang ${sl} --target-lang ${tl} \
	--trainpref $data_dir/train \
	--validpref $data_dir/valid \
	--testpref $data_dir/test \
	--workers 50 \
	--destdir $out_dir/${domain}_${sl}${tl} \
	--joined-dictionary	
	

