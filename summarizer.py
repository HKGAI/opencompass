from mmengine.config import read_base

with read_base():
    from .configs.summarizers.groups.mmlu import mmlu_summary_groups
    from .configs.summarizers.groups.cmmlu import cmmlu_summary_groups
    from .configs.summarizers.groups.ceval import ceval_summary_groups

dataset_abbrs=[
'------- MMLU details -------', 
'mmlu-humanities',
'mmlu-stem',
'mmlu-social-science',
'mmlu-other',
'mmlu',

'---- Standard Benchmarks ---',
'BoolQ',
'piqa',
'siqa',
'hellaswag',
'winogrande',
'ARC-e',
'ARC-c',
'openbookqa_fact',
'commonsense_qa',
'mmlu',

'------ Code Generation -----', #TODO 如何添加0-shot and 3-shot 两种指标
'openai_humaneval',
'mbpp',

'------ World Knowledge -----',#TODO 添加NaturalQuestions 和 两者的0-shot 1-shot 5-shot 64-shot
'nq',
'triviaqa',

'--- Reading Comprehension --',#TODO 添加0-shot 1-shot 4-shot 5-shot 和 QUAC 0-shot 1-shot
'squad2.0',

'---------- Exams -----------',
'math',
'gsm8k',
'TheoremQA',

'--------- Chinese ----------',
"ceval",
'ceval-stem',
'ceval-social-science',
'ceval-humanities',
'ceval-other',
'ceval-hard',

'cmmlu',
'cmmlu-humanities',
'cmmlu-stem',
'cmmlu-social-science',
'cmmlu-other',
'cmmlu-china-specific',

]

summary_groups=sum(
[v for k, v in locals().items() if k.endswith("_summary_groups")], [])
