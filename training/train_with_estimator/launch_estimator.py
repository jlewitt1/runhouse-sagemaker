import os
import runhouse as rh
from sagemaker.pytorch import PyTorch

# https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html#sagemaker.estimator.EstimatorBase
estimator = PyTorch(
    entry_point="train.py",
    # Estimator requires a role ARN (can't be a profile name)
    role="<AWS_ROLE_ARN>",
    # Script can sit anywhere in the file system
    source_dir=os.path.abspath(os.getcwd()),
    # PyTorch version for executing training code
    framework_version="1.13",
    py_version="py39",
    instance_count=1,
    instance_type="ml.m5.large",
    # https://docs.aws.amazon.com/sagemaker/latest/dg/train-warm-pools.html
    keep_alive_period_in_seconds=3600,
    # A list of absolute or relative paths to directories with any additional libraries that
    # should be exported to the cluster
    dependencies=[],
)

if __name__ == "__main__":
    # Launch the training job
    c = rh.sagemaker_cluster(name="rh-sagemaker-estimator", estimator=estimator).up_if_not()
    c.save()

    # To stop the training job:
    # reloaded_cluster.teardown_and_delete()
    # assert not reloaded_cluster.is_up()
