python eval/evaluator.py \
--answer_dir=./results/codegemma-2b/ \
--pattern_path=./eval/pat_list.txt \
--dataset_path=./dataset/question.jsonl \
--model=codegemma \
--passk=1

rm .tmp*