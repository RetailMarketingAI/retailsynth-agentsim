# Maintaining End-to-End Scripts for Agent Training and Evaluation

This folder contains scripts used to train and evaluate the agent. Below is a guide for using these scripts.

## Workflow
1. **Collecting Offline Data:** To collect offline data using a random policy, run:
    ```sh
    python workflow/main.py step=collect_train_data
    ```
2. **Hyperparameter Tuning:** To start a hyperparameter tuning job, run:
    ```sh
    python workflow/optuna_main.py
    ```
    This script initiates an Optuna tuning study and saves the best hyperparameters in `{output_dir}/agent/{agent_type}/tuning_job/tuning_log.pkl`.

3. **Training and Evaluation:**
    - To train an agent with the optimal set of hyperparameters, run:
      ```sh
      python workflow/main.py step=train
      python workflow/main.py step=eval
      ```
    - By default, the script trains the linear Thompson Sampling agent and simulates one evaluation trajectory in `{output_dir}/linear_ts/{agent.name}/`. To train other agents, add the agent config YAML file to `workflow/cfg/agent` and specify the agent argument in the command, e.g.:
      ```sh
      python workflow/main.py step=train agent=linear_ucb
      python workflow/main.py step=eval agent=linear_ucb
      ```

4. **Performance Comparison:** To compare the performance of different agents for all customers and across customer segments, run:
    ```sh
    python workflow/main.py step=segment_customer
    python workflow/main.py step=compute_metrics
    ```

## Setting Up Batch Jobs on AWS

### Prerequisites
Follow these steps to set up batch jobs on AWS if you haven't set up ECR and AWS Batch object yet:

1. **Build Docker Image on EC2 Instance:**
  We recommend to build images on EC2 instances with the same AMI as the AWS Batch environment to avoid compatibility issues.
    ```sh
    docker build -f dockerfile -t latest .
    ```

2. **Authenticate Docker CLI to AWS ECR Registry:**
    ```sh
    aws ecr get-login-password --region {region} | docker login --username AWS --password-stdin {profile_name}.dkr.ecr.{region}.amazonaws.com
    aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 730335247501.dkr.ecr.us-east-1.amazonaws.com
    ```

3. **Tag and Push the Image to AWS ECR:**
    ```sh
    docker tag latest {profile_name}.dkr.ecr.{region}.amazonaws.com/retailsynth:latest
    docker tag latest 730335247501.dkr.ecr.us-east-1.amazonaws.com/retailsynth:latest
    ```
    ```sh
    docker push {profile_name}.dkr.ecr.{region}.amazonaws.com/retailsynth:latest
    docker push 730335247501.dkr.ecr.us-east-1.amazonaws.com/retailsynth:latest
    ```

4. **Create AWS Batch Objects:** If you've already set up job definitions, compute environments, and job queues on AWS Batch, you can skip this step.
    - Start a compute environment:
      ```sh
      aws batch create-compute-environment --cli-input-json file://workflow/aws_batch_template/compute_environment.json
      ```
    - Start a job queue:
      ```sh
      aws batch create-job-queue --cli-input-json file://workflow/aws_batch_template/job_queue.json
      ```
    - Create a job definition:
      ```sh
      aws batch register-job-definition --cli-input-json file://workflow/aws_batch_template/job_definition.json
      ```

    You need to create the following job definitions to proceed with the workflow: `collect_offline_data`, `optuna_tuning`, `train_agent`, `evaluate_agents`.

### Running Batch Jobs
We maintain a shell script `workflow_aws.sh` to run the whole workflow on AWS Batch, including offline data collection, benchmark evaluation, hyperparameter tuning, and agent training & evaluation. The script submits the jobs to the AWS Batch queue and stores the artifacts in the S3 bucket. To run the script, follow these steps:

- To train a specified agent and store artifacts in the S3 bucket:
  ```sh
  sh workflow/workflow_aws.sh
  ```
- To conduct an ablation study or sensitivity study on all supported agents:
  ```sh
  sh workflow/ablation_study.sh
  ```
- To check the status of the submitted jobs:
  ```sh
  aws batch describe-jobs --jobs JOB_ID --output text --query 'jobs[*].status'
  ```
- To check the existence of the output files in the S3 bucket:
  ```sh
  aws s3api head-object --bucket retailsynth-dev --key FILE_NAME
  ```

## Experiment Directory Structure

After each run of an experiment workflow, you can find the experiment artifacts stored in the following structure:

```bash
├── experiment_dir
│   ├── offline_data
│   │   ├── file_uuid_1.pkl
│   │   ├── file_uuid_2.pkl
│   │   └── ...
│   ├── linear_ts
│   │   ├── tuning_log (a data frame containing <params, accumulated revenue mean, accumulated revenue std, training loss>)
│   │   ├── LinTS_optimal
│   │   │   ├── checkpoint/*
│   │   │   ├── eval_result
│   │   │   │   ├── file_uuid_1.pkl
│   │   │   │   ├── file_uuid_2.pkl
│   │   │   │   └── ...
│   │   │   ├── train_loss.pkl
│   │   │   └── agent_cfg.pkl
│   ├── linear_ucb
│   │   ├── tuning_log
│   │   └── ...
│   ├── summary
│   │   ├── all_customer_dfs.pkl
│   │   ├── segmented_dfs.pkl
│   │   └── ...

```

## Result Visualization and Interpretation

After running the workflow, you can visualize the results using the Jupyter notebooks [examples/paper_result_analysis.ipynb](../examples/paper_result_analysis.ipynb). The notebooks provide detailed visualizations and interpretations of the training and evaluation results.