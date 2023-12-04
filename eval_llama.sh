#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate opencompass
nohup python run.py eval_llama_7b_test.py -p slurm_conifg.py&
