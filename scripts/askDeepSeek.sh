GPU=$1

python askHF.py \
--ckpt_path deepseek-ai/deepseek-coder-1.3b-instruct \
--tokenizer_path deepseek-ai/deepseek-coder-1.3b-instruct \
--result_dir ./results/CodeLlama/ \
--shot_number 1 \
--gpu $GPU
