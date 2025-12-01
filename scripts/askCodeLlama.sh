GPU=$1

python askHF.py \
--ckpt_path codellama/CodeLlama-7b-Instruct-hf \
--tokenizer_path codellama/CodeLlama-7b-Instruct-hf \
--result_dir ./results/codellama-7b-instruct/ \
--shot_number 1 \
--gpu $GPU
