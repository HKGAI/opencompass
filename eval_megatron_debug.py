from mmengine.config import read_base
with read_base():
    ########################DATASET##################
    # # Standard Benchmarks
    from .configs.datasets.SuperGLUE_BoolQ.SuperGLUE_BoolQ_ppl import BoolQ_datasets
    from .configs.datasets.piqa.piqa_ppl import piqa_datasets
    from .configs.datasets.siqa.siqa_ppl import siqa_datasets
    from .configs.datasets.hellaswag.hellaswag_ppl import hellaswag_datasets
    from .configs.datasets.winogrande.winogrande_ppl import winogrande_datasets
    from .configs.datasets.ARC_e.ARC_e_ppl import ARC_e_datasets
    from .configs.datasets.ARC_c.ARC_c_ppl import ARC_c_datasets
    from .configs.datasets.obqa.obqa_ppl import obqa_datasets
    # from .configs.datasets.commonsenseqa.commonsenseqa_ppl import commonsenseqa_datasets
    from .configs.datasets.mmlu.mmlu_ppl import mmlu_datasets
    
    # # Code Generation
    # from .configs.datasets.humaneval.humaneval_gen import humaneval_datasets
    # from .configs.datasets.mbpp.mbpp_gen import mbpp_datasets
    # # World Knowledge           need NaturalQuestions
    # from .configs.datasets.nq.nq_gen import nq_datasets
    # from .configs.datasets.triviaqa.triviaqa_gen import triviaqa_datasets
    # from .configs.datasets.truthfulqa.truthfulqa_gen_1e7d8d import truthfulqa_datasets
    # from .configs.datasets.hellaswag.hellaswag_gen_6faab5 import hellaswag_datasets
    
    # # Reading Comprehension       need QUAC
    # from .configs.datasets.squad20.squad20_gen import squad20_datasets

    # # Exams
    # from .configs.datasets.gsm8k.gsm8k_gen import gsm8k_datasets
    # from .configs.datasets.math.math_gen import math_datasets
    # from .configs.datasets.TheoremQA.TheoremQA_gen import TheoremQA_datasets
    # # ceval and cmmlu
    from .configs.datasets.ceval.ceval_ppl import ceval_datasets
    from .configs.datasets.cmmlu.cmmlu_ppl import cmmlu_datasets

    ######################SUMMERIZER#################
    from .summarizer import dataset_abbrs, summary_groups

# 支持多个模型一起测评
# models = [
#     {
#         'type': 'MegatronAPI',
#         'abbr': 'test_megatronapi',
#         'path': 'MegatronAPI',
#         'max_out_len': 100,
#         'max_seq_len': 4096,
#         'batch_size': 32,
#         'run_cfg': {'num_gpus': 1, 'num_procs': 1},
#     },
# ]

models = [
    {
        'type': 'MegatronModel',
        'abbr': f"pt1.5-iter_0228454",
        'ckpt_path': "/workspace/code/checkpoints/hkg_7b_nl_tp1_pp1_mb1_gb1280_gas2/pt1.5/checkpoint",
        'tokenizer_type': "SentencePieceTokenizer",
        'tokenizer_model': "/workspace/code/checkpoints/hkg_7b_nl_tp1_pp1_mb1_gb1280_gas2/pt1.5/hf_ckpt/hkg_baichuan_hf/tokenizer.model", 
        'args_str':  """--use-mcore-models 
                        --num-layers 32 
                        --hidden-size 4096 
                        --ffn-hidden-size 11008 
                        --num-attention-heads 32 
                        --max-position-embeddings 4096 
                        --group-query-attention 
                        --num-query-groups 4 
                        --swiglu 
                        --normalization RMSNorm 
                        --use-rotary-position-embeddings 
                        --untie-embeddings-and-output-weights 
                        --no-position-embedding 
                        --disable-bias-linear 
                        --seed 42 
                        --seq-length 4096 
                        --tensor-model-parallel-size 1 
                        --pipeline-model-parallel-size 1 
                        --expert-model-parallel-size 1 
                        --sequence-parallel""",
        'max_seq_len': 4096,
        'max_out_len': 100,
        'batch_size': 8, # megatron inference只支持fp32读取ckpt，适当降低一些batch_size
        'temperature': 1.0,
        'top_k_sampling': 40,
        'top_p_sampling': 0.0,
        'top_p_decay': 0.0,
        'top_p_bound': 0.0,
        'random_seed': 42,
        'run_cfg': {'num_gpus': 1, 'num_procs': 1},
    },
    {
        'type': 'HuggingFaceCausalLM',
        'abbr': 'pt1.5/1197.76B-hf',
        'path': '/workspace/code/checkpoints/hkg_7b_nl_tp1_pp1_mb1_gb1280_gas2/pt1.5/hf_ckpt/1197.76B',
        'tokenizer_path': '/workspace/code/checkpoints/hkg_7b_nl_tp1_pp1_mb1_gb1280_gas2/pt1.5/hf_ckpt/hkg_baichuan_hf',
        'tokenizer_kwargs': {
            'padding_side': 'left',
            'truncation_side': 'left',
            'use_fast': False,
            'trust_remote_code': True,
        },
        'max_out_len': 100,
        'max_seq_len': 4096,
        'batch_size': 16,
        'model_kwargs': {
            'device_map': 'auto', 
            'trust_remote_code': True,
        },
        'batch_padding': False,
        'run_cfg': {'num_gpus': 1, 'num_procs': 1},
    },
]

# models = [
#     {
#         'type': 'MegatronModel',
#         'abbr': f"exp5.1-iter_0202640",
#         'ckpt_path': "/workspace/code/checkpoints/hkg_7b_nl_tp2_pp1_mb1_gb1024_gas8/exp5.1/checkpoint",
#         'tokenizer_type': "SentencePieceTokenizer",
#         'tokenizer_model': "/workspace/code/checkpoints/hkg_7b_nl_tp1_pp1_mb1_gb1280_gas2/pt1.5/hf_ckpt/hkg_baichuan_hf/tokenizer.model", 
#         'args_str':  """--use-mcore-models 
#                         --num-layers 32 
#                         --hidden-size 4096 
#                         --ffn-hidden-size 11008 
#                         --num-attention-heads 32 
#                         --max-position-embeddings 4096 
#                         --group-query-attention 
#                         --num-query-groups 4 
#                         --swiglu 
#                         --normalization RMSNorm 
#                         --use-rotary-position-embeddings 
#                         --untie-embeddings-and-output-weights 
#                         --add_position_embedding False
#                         --no-position-embedding 
#                         --disable-bias-linear 
#                         --seed 42 
#                         --seq-length 4096 
#                         --tensor-model-parallel-size 2
#                         --pipeline-model-parallel-size 1 
#                         --expert-model-parallel-size 1 
#                         --sequence-parallel""",
#         'max_seq_len': 4096,
#         'max_out_len': 100,
#         'batch_size': 8, # megatron inference只支持fp32读取ckpt，适当降低一些batch_size
#         'temperature': 1.0,
#         'top_k_sampling': 40,
#         'top_p_sampling': 0.0,
#         'top_p_decay': 0.0,
#         'top_p_bound': 0.0,
#         'random_seed': 42,
#         'run_cfg': {'num_gpus': 1, 'num_procs': 1},
#     },
# ]

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
summarizer = dict(dataset_abbrs=dataset_abbrs, summary_groups=summary_groups, )
