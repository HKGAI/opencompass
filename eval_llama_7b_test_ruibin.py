from mmengine.config import read_base

with read_base():
    ########################DATASET##################
    # Standard Benchmarks
    from .configs.datasets.hellaswag.hellaswag_ppl_a6e128 import hellaswag_datasets
    # from .configs.datasets.mmlu.mmlu_ppl_ac766d import mmlu_datasets

    #########################MODEL###################
    from .hf_llama_7b_ruibin import models

    ######################SUMMERIZER#################
    from .summarizer import dataset_abbrs, summary_groups

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

models = models

summarizer = dict(dataset_abbrs=dataset_abbrs, summary_groups=summary_groups, )