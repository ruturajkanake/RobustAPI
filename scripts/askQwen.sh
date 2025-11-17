GPU=$1

python askHF.py \
--ckpt_path Qwen/Qwen2.5-Coder-1.5B-Instruct \
--tokenizer_path Qwen/Qwen2.5-Coder-1.5B-Instruct \
--result_dir ./results/qwen2.5-coder-1.5/ \
--shot_number 1 \
--gpu $GPU
