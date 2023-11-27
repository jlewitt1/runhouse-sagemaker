# 🏃‍♀️Runhouse🏠 & SageMaker

## ❣️ Why do we love SageMaker?

* **Serverless compute**: SageMaker provides a more scalable experience than EC2, which means you don’t need to 
be responsible for auto-stopping, scheduling, or worry about accessing compute in a K8s cluster and managing queueing 
jobs or running them in parallel. With SageMaker you can easily launch multiple instances at the same time.  

* **Launching with containers**: SageMaker allows you to launch a cluster with a docker container. This gives you a 
more K8s like experience of launching compute with a lightweight image rather than an 
[AMI](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AMIs.html), which is more difficult to publish and 
expensive to maintain.

* **More reliable scaling**: Instead of maintaining a fleet of GPUs with some orchestrator like ECS, 
SageMaker allows you to trigger many jobs in parallel. If you have training or batch jobs which are triggered by 
customers, SageMaker is a much more reliable way to maintain those jobs at scale. It's also very easy to mix and match 
SageMaker with other compute options within your existing fleet

* **GPU availability**: We've observed that GPUs tend to be more available on SageMaker compared to EC2.

## 🤝 Using Runhouse with SageMaker

[Runhouse](https://www.run.house) makes it super easy to get started with SageMaker or continue to use SageMaker within your existing stack. 
Whether you are hacking on a single CPU / GPU instance or deploying a fleet of clusters to support training, 
inference, or hyperparameter tuning, SageMaker is an ideal platform to handle all of these needs and reliably scale 
as your ML stack evolves. Using the Runhouse uniform API allows you to transition from single box to multi-box to 
a pool of compute without the need for any code migrations or refactors.

The Runhouse [SageMakerCluster](https://www.run.house/docs/main/en/api/python/cluster#sagemakercluster-class) 
makes the process of onboarding to SageMaker more smooth, saving you the need to create estimators, or conform the 
code to the SageMaker APIs. This translation step can take anywhere from days to months, and leads to rampant code 
duplication, forking, versioning and lineage issues.

## 🚀 Getting Started

SageMaker clusters require AWS CLI V2 and configuring the SageMaker IAM role with the AWS Systems Manager.

In order to launch a cluster, you must grant SageMaker the necessary permissions with an IAM role, 
which can be provided either by name, full ARN or with a profile name. You can also specify a profile explicitly or with the 
`AWS_PROFILE` environment variable.

The examples in this repo use an AWS profile name, which Runhouse extracts from the local `~/.aws/config` file. 
If your config file contains the below profile: 

```ini
[profile sagemaker]
role_arn = arn:aws:iam::123456789:role/service-role/AmazonSageMaker-ExecutionRole-20230717T192142
region = us-east-1
source_profile = default
```

You can then pass in `profile=sagemaker` when initializing the cluster.

For a more detailed walkthrough, see the
[SageMaker Hardware Setup](https://www.run.house/docs/stable/en/api/python/cluster#sagemaker-hardware-setup) section of the Runhouse docs.

## 🛣️ Core Usage Paths

This repo contains examples highlighting some common SageMaker use cases: 

### Inference

Runhouse facilitates easier access to the SageMaker compute from different environments. 
You can interact with the compute from notebooks, IDEs, research, pipeline DAG, or any python interpreter. 
Runhouse allows you to SSH directly onto the cluster, update or suspend cluster autostop, and stream logs 
directly from the cluster. 

We've highlighted to inference examples:
- **[Stable Diffusion](inference/stable_diffusion.py)**: Create an inference service which receives a prompt input text and outputs a PIL image
- **[Llama2](inference/llama2inference.py)**: Stand up an inference service using the Hugging Face chat model (Note: this requires a token)

Running the SD inference example:

```bash
python inference/stable_diffusion.py
```

Running the Llama2 inference example:

```bash
python inference/llama2inference.py
```

### Training

We'll use a simple PyTorch model to illustrate the different ways we can run training on SageMaker compute via Runhouse.
In each of these examples, Runhouse is responsible for spinning up the requested SageMaker compute, and executing the 
training code on the cluster.

(1) [**Simple train**](training/simple_train): Use Runhouse to create the SageMaker cluster and handle running the 
training code. In this example, we wrap the training code in a Runhouse function which we send to our cluster for 
execution. The changes to the source code are minimal - we simply instantiate our SageMaker cluster, wrap the training 
code in a function, and then call it in the same way we would call a local function.

Run this example:

```bash
python training/simple_train/train.py
```

```bash
python training/simple_train/train.py --epochs 5 --learning-rate 0.001 --batch-size 64
```

(2) [**Interactive train**](training/interactive_train): Convert the training code into a 
Runhouse [Module](https://www.run.house/docs/api/python/module) class, with separate methods for training, eval, and 
inference. While this requires slightly more modifications to the original source code, it gives us a stateful and 
interactive experience with the model on the cluster, as if we are in a notebook environment. We can much more easily 
run training epochs or try out the most recent checkpoint of the model that's been saved, without the need for 
packaging up the model and deploying it to a separate endpoint.

Run this example:

```bash
python training/interactive_train/train.py
```

```bash
python training/interactive_train/train.py --output-path ~/.cache/models/output
```

🦸 Both of these examples unlock a key superpower - the ability to easily run class methods on a remote cluster, 
**without** needing to translate or migrate the code onto another system.

(3) [**Train with Estimator**](training/train_with_estimator): Use the SageMaker SDK to create an
[estimator](https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html). This is useful if you are already 
using the SageMaker APIs. In this example, we define a SageMaker estimator which loads in the training code from 
a separate file where the training code lives.

*Note*: Logs for the training job can be viewed on the cluster in path: `/opt/ml/code/sm_cluster.out` or in 
AWS Cloudwatch in the default log group folder (e.g. `/aws/sagemaker/TrainingJobs`)

### Hyperparameter Tuning (⚠️ Under active development)

For this [example](hyperparameter_tuning/hp_tuning.py), we use [Ray Tune](https://docs.ray.io/en/latest/tune/index.html) to try different hyperparameter 
combinations on SageMaker compute. 

## 👨‍🏫 Resources
[**Docs**](https://www.run.house/docs/api/python/cluster#sagemakercluster-class):
High-level overviews of the architecture, detailed API references, and basic API examples for the SageMaker 
integration.

**Blog**: Coming soon... 

