GPU=$1

python askHF.py \
--ckpt_path microsoft/Phi-3-mini-4k-instruct \
--tokenizer_path microsoft/Phi-3-mini-4k-instruct \
--result_dir ./results/phi3-mini-4k/ \
--shot_number 1 \
--gpu $GPU
