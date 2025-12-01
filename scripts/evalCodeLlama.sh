python eval/evaluator.py \
--answer_dir=./results/codellama-7b-instruct/ \
--pattern_path=./eval/pat_list.txt \
--dataset_path=./dataset/question.jsonl \
--model=llama \
--passk=1

rm .tmp*