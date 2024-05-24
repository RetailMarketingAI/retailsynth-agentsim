#!/bin/bash
N_OFFLINE_DATA_TRAJ=2
N_ARM=2
N_TRAIN_JOB=2
N_LINEAR_TRAIN_BATCH_SIZE=2
N_LINEAR_TRAIN_EPOCH=1
N_NN_TRAIN_BATCH_SIZE=2
N_NN_TRAIN_EPOCH=2
N_EVAL_TRAJ=2

# ####
# # COLLECT OFFLINE DATA
# ####

echo "Collecting offline data."

python workflow/main.py hydra/launcher=joblib experiment.storage=local step=collect_train_data +task=range\(${N_OFFLINE_DATA_TRAJ}\) -m

echo "Finished collecting offline data."

####
# COLLECT EVALUATION TRAJECTORIES FOR BENCHMARK
####
echo "Collect evaluation trajectories for benchmark agents."

run_train_eval() {
    python workflow/main.py step=train experiment.storage=local experiment.run_id=range\(${N_TRAIN_JOB}\) $@ -m
    python workflow/main.py hydra/launcher=joblib step=eval experiment.storage=local experiment.instance_eval_id=range\(${N_EVAL_TRAJ}\) $@ -m 

}

echo "Collecting evaluation trajectories for benchmark fixed arm agents."

run_train_eval agent=fixed_arm agent.agent_params.chosen_arm=range\(${N_ARM}\) train.read_params_from_tuning_log=False

echo "Evaluating random agent."
run_train_eval agent=random train.read_params_from_tuning_log=False

echo "Finished collecting evaluation trajectories for benchmark agents." 

# ####
# # SKIP HYPERPARAMETER TUNING IN THE LOCAL SCRIPT
# ####
echo "Skipping hyperparameter tuning."

# ####
# # TRAIN BANDIT AGENT
# ####

echo "Training linear ucb agents."
run_train_eval agent=linear_ucb agent.train_params.batch_size=$N_LINEAR_TRAIN_BATCH_SIZE agent.train_params.n_train_epoch=$N_LINEAR_TRAIN_EPOCH

echo "Training linear ts agents."
run_train_eval agent=linear_ts agent.train_params.batch_size=$N_LINEAR_TRAIN_BATCH_SIZE agent.train_params.n_train_epoch=$N_LINEAR_TRAIN_EPOCH

echo "Train neural boltzmann agents."
run_train_eval agent=neural_boltzmann agent.train_params.batch_size=$N_LINEAR_TRAIN_BATCH_SIZE agent.train_params.n_train_epoch=$N_LINEAR_TRAIN_EPOCH

echo "Finished training bandit agents."

####
# TRAIN RL AGENT
####

echo "Training dqn agents."
run_train_eval agent=dqn agent.train_params.batch_size=$N_NN_TRAIN_BATCH_SIZE agent.train_params.n_train_epoch=$N_NN_TRAIN_EPOCH

echo "Training ppo agents."
run_train_eval agent=ppo agent.train_params.batch_size=$N_NN_TRAIN_BATCH_SIZE agent.train_params.n_train_epoch=$N_NN_TRAIN_EPOCH

echo "Finished training rl agents."

####
# COMPUTE EVALUATION METRICS
####
echo "Computing evaluation metrics."

python workflow/main.py step=segment_customer experiment.storage=local
python workflow/main.py step=compute_metrics experiment.storage=local
echo "Finished computing evaluation metrics."

####
# Convert sampled offline data trajectories to a pandas data frame for feature selection.
####
echo "Collecting a dataframe for feature selection."

python workflow/main.py step=feature_selection experiment.storage=local
echo "Finished collecting feature selection data."
