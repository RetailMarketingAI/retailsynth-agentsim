# DEFINE GLOBAL VARIABLES
SCENARIO='test'
ENV='3c10'
N_ARM=3
N_OFFLINE_DATA_TRAJ=2 # This value should be at least 2
N_TRAIN_JOB=1
CONFIG_PATH='../tests/cfg'
# for parameters like training batch size, training epoch, length of the training trajectory and evaluation 
# trajectory, please refer to `cfg` folder for the default values.

# To expedite the debugging process, we can
# 1. use configuration used for running pytest, set `CONFIG_PATH='../tests/cfg'`
# 2. go to the `cfg/synthetic_data` folder, and lower down the values for n_customer, n_product, n_category, category_product_count
# 3. goto the `cfg/agent` folder, and lower down the values for train_params.batch_size, train_params.n_train_epoch
# 4. go to the `cfg/optuna` folder, and lower down the values for n_trial, n_job_in_parallel, n_instance_per_trial

N_EVAL_TRAJ=2 # This value should be at least 2
AGENT='linear_ucb' # Other options, with workflow/cfg, include: linear_ts, neural_boltzmann, dqn, ppo
EVALUATE_AGENT_NAME='["LinUCB9f"]' # Other options, with workflow/cfg, include: LinTS9f, NeuralBoltzmann9f, DQN9f, PPO9f
BENCHMARK_AGENTS="""["FixedArm_0", "FixedArm_1", "FixedArm_2", "random_agent"]"""

RUN_OFFLINE_DATA=true
RUN_BENCHMARK_EVALUATION=true
RUN_HYPERPARAMETER_TUNING=true
RUN_TRAINING=true
RUN_RESULT_CHECK=true

##############################################################################
# COLLECT OFFLINE DATA
if ${RUN_OFFLINE_DATA}; then

    job_id_offline_data=$(aws batch submit-job \
        --job-name explore_random_policy \
        --job-queue experiment_workflow \
        --job-definition collect_offline_data \
        --array-properties size=${N_OFFLINE_DATA_TRAJ} \
        --container-overrides command="['python', 'workflow/main.py', '--config-path=$CONFIG_PATH', 'step=collect_train_data', 'synthetic_data=${SCENARIO}', 'pyenv=${ENV}', 'train.offline_data_policy.type=random']" \
        --output text --query jobId)
    echo "Job ID for collecting offline trajectory using random policy: ${job_id_1}"

else
    job_id_offline_data="None"
    echo "Skip collecting offline data"
fi

#############################################################################
# COLLECT BENCHMARK EVALUATION TRAJECTORIES
if ${RUN_BENCHMARK_EVALUATION}; then
    eval_job_ids=()
    for ((arm=0;arm<${N_ARM};arm++)); do
        for ((id=0;id<${N_TRAIN_JOB};id++)); do

            job_id=$(aws batch submit-job \
                --job-name "${SCENARIO}_run_${arm}" \
                --job-queue experiment_workflow \
                --job-definition collect_offline_data \
                --depends-on jobId=${job_id_offline_data} \
                --container-overrides command="['python', 'workflow/main.py', '--config-path=$CONFIG_PATH', 'step=train', 'synthetic_data=${SCENARIO}', 'pyenv=${ENV}', 'agent=fixed_arm', 'agent.agent_params.chosen_arm=${arm}', \"experiment.run_id='${id}'\"]" \
                --output text --query jobId)

            job_id=$(aws batch submit-job \
                --job-name "${SCENARIO}_collect_${arm}_eval_traj" \
                --job-queue experiment_workflow \
                --job-definition collect_offline_data \
                --array-properties size=${N_EVAL_TRAJ} \
                --depends-on jobId=${job_id} \
                --container-overrides command="['python', 'workflow/main.py', '--config-path=$CONFIG_PATH', 'step=eval', 'synthetic_data=${SCENARIO}', 'pyenv=${ENV}', 'agent=fixed_arm', 'agent.agent_params.chosen_arm=${arm}', \"experiment.run_id='${id}'\"]" \
                --output text --query jobId)

            eval_job_ids+=("${job_id}")
        
        done

        echo "job_id for arm ${arm}: ${job_id}"
    done


    for ((id=0;id<${N_TRAIN_JOB};id++)); do
        job_id=$(aws batch submit-job \
            --job-name "${SCENARIO}_run_random" \
            --job-queue experiment_workflow \
            --job-definition train_agent \
            --depends-on jobId=${job_id_offline_data} \
            --container-overrides command="['python', 'workflow/main.py', '--config-path=$CONFIG_PATH', 'step=train', 'synthetic_data=${SCENARIO}', 'pyenv=${ENV}', 'agent=random', \"experiment.run_id='${id}'\"]" \
            --output text --query jobId)

        job_id=$(aws batch submit-job \
            --job-name "${SCENARIO}_collect_random_eval_traj" \
            --job-queue experiment_workflow \
            --job-definition collect_offline_data \
            --array-properties size=${N_EVAL_TRAJ} \
            --depends-on jobId=${job_id} \
            --container-overrides command="['python', 'workflow/main.py', '--config-path=$CONFIG_PATH', 'step=eval', 'synthetic_data=${SCENARIO}', 'pyenv=${ENV}', 'agent=random', \"experiment.run_id='${id}'\"]" \
            --output text --query jobId)

        eval_job_ids+=("${job_id}")
        echo "job_id for random: ${job_id}"
    done

    depends_on=" --depends-on "
    for id in "${eval_job_ids[@]}"; do
        depends_on+="jobId=${id} "
    done

    job_id=$(aws batch submit-job \
        --job-name "${SCENARIO}_segment_customers" \
        --job-queue experiment_workflow \
        --job-definition evaluate_agents \
        ${depends_on} \
        --container-overrides command="""['python', 'workflow/main.py', '--config-path=$CONFIG_PATH', 'step=segment_customer', 'synthetic_data=${SCENARIO}', 'pyenv=${ENV}']""" \
        --output text --query jobId)
    job_id_benchmark_evaluation=$(aws batch submit-job \
        --job-name "${SCENARIO}_evaluate_agents" \
        --job-queue experiment_workflow \
        --job-definition evaluate_agents \
        --depends-on jobId=${job_id} \
        --container-overrides command="""['python', 'workflow/main.py', '--config-path=$CONFIG_PATH', 'step=compute_metrics', 'synthetic_data=${SCENARIO}', 'pyenv=${ENV}', 'summary.agents=${BENCHMARK_AGENTS}', 'summary.summary_file_prefix=benchmark_']""" \
        --output text --query jobId)

    echo "job_id for evaluating benchmark agents: ${job_id_benchmark_evaluation}"

else
    job_id_benchmark_evaluation="None"
    echo "Skip collecting benchmark evaluation trajectories"
fi
#############################################################################
# RUN HYPERPARAMETER TUNING FOR GIVEN AGENT
if ${RUN_HYPERPARAMETER_TUNING}; then
    job_id_hyperparm=$(aws batch submit-job \
        --job-name "${SCENARIO}_tuning_${AGENT}" \
        --job-queue experiment_workflow \
        --job-definition optuna_tuning \
        --depends-on jobId=${job_id_offline_data} \
        --container-overrides command="['python', 'workflow/optuna_main.py', '--config-path=$CONFIG_PATH', 'synthetic_data=${SCENARIO}', 'pyenv=${ENV}', 'optuna=${AGENT}']" \
        --output text --query jobId)

    echo "job_id for hyperparameter tuning ${AGENT}: ${job_id_hyperparm}"
else
    job_id_hyperparm="None"
    echo "Skip hyperparameter tuning for ${AGENT}"
fi

#############################################################################
# TRAIN AGENT
if ${RUN_TRAINING}; then
    train_job_ids=()
    for ((id=0;id<${N_TRAIN_JOB};id++)); do
        job_id=$(aws batch submit-job \
            --job-name "${SCENARIO}_train_${AGENT}" \
            --job-queue experiment_workflow \
            --job-definition train_agent \
            --depends-on jobId=${job_id_hyperparm} jobId=${job_id_offline_data} \
            --container-overrides command="""[\"python\", \"workflow/main.py\", \"--config-path=$CONFIG_PATH\", \"step=train\", \"train.read_params_from_tuning_log=True\", \"experiment.run_id='${id}'\"]""" \
            --output text --query jobId)

        job_id=$(aws batch submit-job \
            --job-name "${SCENARIO}_collect_${AGENT}_eval_traj" \
            --job-queue experiment_workflow \
            --job-definition collect_offline_data \
            --array-properties size=${N_EVAL_TRAJ} \
            --depends-on jobId=${job_id} \
            --container-overrides command="""[\"python\", \"workflow/main.py\", \"--config-path=$CONFIG_PATH\", \"step=eval\", \"experiment.run_id='${id}'\"]""" \
            --output text --query jobId)

        train_job_ids+=(${job_id})

        echo "job_id for ${AGENT} ${id}: ${job_id}"
    done

    depends_on=" --depends-on "
    for id in "${train_job_ids[@]}"; do
        depends_on+="jobId=${id} "
    done

    job_id_agent_evaluation=$(aws batch submit-job \
        --job-name "${SCENARIO}_evaluate_agents" \
        --job-queue experiment_workflow \
        --job-definition evaluate_agents \
        ${depends_on} \
        --container-overrides command="""['python', 'workflow/main.py', '--config-path=$CONFIG_PATH', 'step=compute_metrics', 'synthetic_data=${SCENARIO}', 'pyenv=${ENV}', 'summary.agents=${EVALUATE_AGENT_NAME}', 'summary.summary_file_prefix=${AGENT}_']""" \
        --output text --query jobId)

    echo "job_id for evaluating ${AGENT}: ${job_id_agent_evaluation}"

else
    job_id_agent_evaluation="None"
    echo "Skip training ${AGENT}"
fi


# #############################################################################
# CHECK IF RESULTS ARE READY
if ${RUN_RESULT_CHECK}; then
    echo "Check the result path for the evaluation metrics."

    reponse=$(aws batch describe-jobs --jobs $job_id_benchmark_evaluation --output text --query 'jobs[*].status')
    echo "Job status for benchmark evaluation: ${reponse}"
    benchmark_summary_path="${ENV}_${SCENARIO}_100customers_20products/summary/benchmark_all_customer_dfs.pkl"
    aws s3api head-object --bucket retailsynth-dev --key $benchmark_summary_path

    reponse=$(aws batch describe-jobs --jobs $job_id_agent_evaluation --output text --query 'jobs[*].status')
    echo "Job status for ${AGENT} evaluation: ${reponse}"
    linear_ucb_summary_path="${ENV}_${SCENARIO}_100customers_20products/summary/linear_ucb_all_customer_dfs.pkl"
    aws s3api head-object --bucket retailsynth-dev --key $linear_ucb_summary_path

else
    echo "Skip checking the result path for the evaluation metrics."
fi