#!/bin/bash
nohup python run.py --datasets  mmlu_ppl \
--hf-path LLM360/Amber \
--model-kwargs device_map='auto' \
--tokenizer-kwargs padding_side='left' truncation='left' trust_remote_code=True \
--max-seq-len 2048 \
--max-out-len 100 \
--batch-size 16  \
--num-gpus 1 -r > test_amber_log 2>&1

# nohup python run.py eval_llama_7b_test_new.py -r > 301.99B_log 2>&1