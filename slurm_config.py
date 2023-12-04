from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import SlurmRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=5000),
    runner=dict(
        type=SlurmRunner,
        max_num_workers=64,
        task=dict(type=OpenICLInferTask),
        retry=5),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=32,
        task=dict(type=OpenICLEvalTask)),
)