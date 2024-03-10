from mmengine.config import read_base
from opencompass.models import HuggingFaceCausalLM

models = [
    {
        'type': 'HuggingFaceCausalLM',
        'abbr': '69.99B',
        'path': '/workspace/megatron/checkpoints/hkg_7b_nl_tp1_pp1_mb1_gb1280_gas2/pt1.4/hf_ckpt/69.99B',
        'tokenizer_path': '/workspace/megatron/checkpoints/hkg_7b_nl_tp1_pp1_mb1_gb1024_gas2/pt1.2/hf_ckpt/hkg_baichuan_hf',
        'tokenizer_kwargs': {
            'padding_side': 'left',
            'truncation_side': 'left',
            'use_fast': False,
            'trust_remote_code': True,
        },
        'max_out_len': 100,
        'max_seq_len': 4096,
        'batch_size': 4,
        'model_kwargs': {
            'device_map': 'auto', 
            'trust_remote_code': True,
        },
        'batch_padding': False,
        'run_cfg': {'num_gpus': 1, 'num_procs': 1},
    },
]


with read_base():
    ########################DATASET##################
    # Standard Benchmarks
    from .configs.datasets.hellaswag.hellaswag_gen_6faab5 import hellaswag_datasets
    from .configs.datasets.mmlu.mmlu_gen_5d1409 import mmlu_datasets
    # from .configs.datasets.hellaswag.hellaswag_ppl_a6e128 import hellaswag_datasets
    # from .configs.datasets.mmlu.mmlu_ppl_ac766d import mmlu_datasets
    
    ######################SUMMERIZER#################
    from .configs.summarizers.groups.mmlu import mmlu_summary_groups

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

summarizer = dict(dataset_abbrs=['hellaswag'], summary_groups=mmlu_summary_groups, )
