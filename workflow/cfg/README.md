# Config Folder Layout
This repository's config folder organizes various configuration files and directories to streamline the setup and management of different components and environments. Below is a breakdown of the folder structure:

## agent
The `agent` directory holds multiple YAML files, each containing configurations for a different agent module. We provide support for various agents, such as LinTS, LinUCB, and DQN agents.

## cloud
The `cloud` directory contains configuration files related to setting up the AWS environment. These files define settings such as access keys, regions, and other parameters required for interacting with AWS services.

## pyenv
The `pyenv` directory encompasses configuration files specific to the Python  bandit environment used for TensorFlow (TF) learning. It includes settings for number of arms, coupon levels, and other configurations related to the TF learning environment.

## synthetic_data
In the `synthetic_data` directory, you'll find configuration files for the data synthesizer module. These files define parameters and settings for generating synthetic data used in training and evaluating scenarios.

## optuna 
The `optuna` directory contains configuration files for the Optuna hyperparameter tuning process. These files define the search space, number of trials, objective function, AWS Batch job setup, and other parameters required for the tuning process.

## hydra/job_logging
The `job_logging` directory contains configuration files for logging setup. These files define the logging level, format, and other parameters required for logging job information.


Additionally, the repository includes a specific configuration file named   `agent_training.yaml`. This file consolidates values and configurations for each of the above items, providing a centralized location for managing the overall training setup for the agent module.
