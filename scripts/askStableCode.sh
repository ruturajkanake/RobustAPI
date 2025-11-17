GPU=$1

python askHF.py \
--ckpt_path stabilityai/stable-code-3b \
--tokenizer_path stabilityai/stable-code-3b \
--result_dir ./results/stable-code-3b/ \
--shot_number 1 \
--gpu $GPU
