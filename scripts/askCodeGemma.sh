GPU=$1

python askHF.py \
--ckpt_path google/codegemma-2b \
--tokenizer_path google/codegemma-2b \
--result_dir ./results/codegemma-2b/ \
--shot_number 1 \
--gpu $GPU
