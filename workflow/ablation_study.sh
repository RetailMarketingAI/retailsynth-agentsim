# DEFINE GLOBAL VARIABLES

FEATURES="[ \
'avg_purchase_price', \
'avg_purchase_discount', \
'avg_purchase_probability', \
'avg_purchase_quantity', \
'previous_transaction', \
'avg_redeemed_discount', \
'product_price', \
'marketing_feature', \
'observed_customer_product_feature']"
N_FEATURE=9
SCENARIO='no_marketing'
ENV='11c50'
N_TRAIN_JOB=10
TRAIN_JOB_START_IDX=0
N_TRAIN_SIZE=1000
EVAL_SIZE=100
LINEAR_BANDIT_TRAIN_EPOCH=2
LINEAR_BANDIT_TRAIN_BATCH=500
EARLY_STOPPING_PATIENCE=15

RUN_LINEAR_UCB=true
RUN_LINEAR_TS=true
RUN_NB=false
RUN_DQN=false
RUN_PPO=false

# This script assumes the following steps are finished:
# 1. Offline data is collected and stored in s3 bucket
# 2. Customer segments is labeled and stored in s3 bucket
# 3. Hyperparameter tuning is finished and the best hyperparameters are extracted for the following agents.

# #################################################################################################################################################
if ${RUN_LINEAR_UCB}; then
    echo "TRAIN LINEAR UCB AGENTS"

    PARAM_OVERWRITE=(
        "agent.agent_params.alpha=0.8483317906758103"
        "agent.agent_params.gamma=0.10880708476170549"
        "agent.agent_params.name=LinUCB${N_FEATURE}f_${N_TRAIN_SIZE}"
        "train.read_params_from_tuning_log=False"
        "train.train_data_size=${N_TRAIN_SIZE}"
        "agent.train_params.batch_size=${LINEAR_BANDIT_TRAIN_BATCH}"
        "agent.train_params.n_train_epoch=${LINEAR_BANDIT_TRAIN_EPOCH}"
    )
    AGENT='linear_ucb'

    command_overwrite="""\"synthetic_data=${SCENARIO}\", \"pyenv=${ENV}\", \"agent=${AGENT}\", \"agent.agent_params.context_features=${FEATURES}\""""
    for param in "${PARAM_OVERWRITE[@]}"; do
        command_overwrite+=", \"${param}\""
    done
    echo ${command_overwrite}

    train_job_ids=()
    for ((id=${TRAIN_JOB_START_IDX=0};id<${N_TRAIN_JOB};id++)); do
        job_id=$(aws batch submit-job \
            --job-name "${SCENARIO}_train_${AGENT}" \
            --job-queue experiment_workflow \
            --job-definition train_agent \
            --container-overrides command="""[\"python\", \"workflow/main.py\", \"step=train\", ${command_overwrite}, \"experiment.run_id='${id}'\"]""" \
            --output text --query jobId)

        job_id=$(aws batch submit-job \
            --job-name "${SCENARIO}_collect_${AGENT}_eval_traj" \
            --job-queue experiment_workflow \
            --job-definition collect_offline_data \
            --array-properties size=${EVAL_SIZE} \
            --depends-on jobId=${job_id} \
            --container-overrides command="""[\"python\", \"workflow/main.py\", \"step=eval\", ${command_overwrite}, \"experiment.run_id='${id}'\"]""" \
            --output text --query jobId)

        train_job_ids+=(${job_id})
        echo "job_id for run ${id}: ${job_id}"
    done


    AGENT_NAME="LinUCB${N_FEATURE}f_${N_TRAIN_SIZE}"
    EVALUATE_AGENTS="[\"${AGENT_NAME}\"]"

    depends_on=" --depends-on "
    for id in "${train_job_ids[@]}"; do
        depends_on+="jobId=${id} "
    done

    job_id_4=$(aws batch submit-job \
        --job-name "${SCENARIO}_evaluate_agents_${AGENT_NAME}" \
        --job-queue experiment_workflow \
        --job-definition evaluate_agents \
        ${depends_on} \
        --container-overrides command="""['python', 'workflow/main.py', 'step=compute_metrics', 'synthetic_data=${SCENARIO}', 'pyenv=${ENV}', 'summary.agents=${EVALUATE_AGENTS}', 'summary.summary_file_prefix=${AGENT_NAME}_']""" \
        --output text --query jobId)

fi

# # #################################################################################################################################################
if ${RUN_LINEAR_TS}; then
    echo "TRAIN LINEAR TS AGENTS"

    PARAM_OVERWRITE=(
        "agent.agent_params.alpha=0.738665395877725"
        "agent.agent_params.gamma=0.8119218938166818"
        "agent.agent_params.name=LinTS${N_FEATURE}f_${N_TRAIN_SIZE}"
        "train.read_params_from_tuning_log=False"
        "train.train_data_size=${N_TRAIN_SIZE}"
        "agent.train_params.batch_size=${LINEAR_BANDIT_TRAIN_BATCH}"
        "agent.train_params.n_train_epoch=${LINEAR_BANDIT_TRAIN_EPOCH}"
    )
    AGENT='linear_ts'

    command_overwrite="""\"synthetic_data=${SCENARIO}\", \"pyenv=${ENV}\", \"agent=${AGENT}\", \"agent.agent_params.context_features=${FEATURES}\""""
    for param in "${PARAM_OVERWRITE[@]}"; do
        command_overwrite+=", \"${param}\""
    done
    echo ${command_overwrite}

    train_job_ids=()
    for ((id=${TRAIN_JOB_START_IDX=0};id<${N_TRAIN_JOB};id++)); do
        job_id=$(aws batch submit-job \
            --job-name "${SCENARIO}_train_${AGENT}" \
            --job-queue experiment_workflow \
            --job-definition train_agent \
            --container-overrides command="""[\"python\", \"workflow/main.py\", \"step=train\", ${command_overwrite}, \"experiment.run_id='${id}'\"]""" \
            --output text --query jobId)

        job_id=$(aws batch submit-job \
            --job-name "${SCENARIO}_collect_${AGENT}_eval_traj" \
            --job-queue experiment_workflow \
            --job-definition collect_offline_data \
            --array-properties size=${EVAL_SIZE} \
            --depends-on jobId=${job_id} \
            --container-overrides command="""[\"python\", \"workflow/main.py\", \"step=eval\", ${command_overwrite}, \"experiment.run_id='${id}'\"]""" \
            --output text --query jobId)

        train_job_ids+=(${job_id})
        echo "job_id for run ${id}: ${job_id}"
    done


    AGENT_NAME="LinTS${N_FEATURE}f_${N_TRAIN_SIZE}"
    EVALUATE_AGENTS="[\"${AGENT_NAME}\"]"
    depends_on=" --depends-on "
    for id in "${train_job_ids[@]}"; do
        depends_on+="jobId=${id} "
    done

    job_id_4=$(aws batch submit-job \
        --job-name "${SCENARIO}_evaluate_agents_${AGENT_NAME}" \
        --job-queue experiment_workflow \
        --job-definition evaluate_agents \
        ${depends_on} \
        --container-overrides command="""['python', 'workflow/main.py', 'step=compute_metrics', 'synthetic_data=${SCENARIO}', 'pyenv=${ENV}', 'summary.agents=${EVALUATE_AGENTS}', 'summary.summary_file_prefix=${AGENT_NAME}_']""" \
        --output text --query jobId)
fi

# #################################################################################################################################################
if ${RUN_NB}; then
    echo "TRAIN NEURAL BOLTZMANN AGENTS"

    PARAM_OVERWRITE=(
        "agent.agent_params.optimizer.learning_rate=0.02496721629015512"
        "agent.agent_params.temperature=0.16645228911819948"
        "agent.agent_params.hidden_layers=[8, 2]"
        "agent.agent_params.name=NeuralBoltzmann${N_FEATURE}f_${N_TRAIN_SIZE}"
        "train.read_params_from_tuning_log=False"
        "train.train_data_size=${N_TRAIN_SIZE}"
    )
    AGENT='neural_boltzmann'

    command_overwrite="""\"synthetic_data=${SCENARIO}\", \"pyenv=${ENV}\", \"agent=${AGENT}\", \"agent.agent_params.context_features=${FEATURES}\""""
    for param in "${PARAM_OVERWRITE[@]}"; do
        command_overwrite+=", \"${param}\""
    done
    echo ${command_overwrite}

    train_job_ids=()
    for ((id=${TRAIN_JOB_START_IDX=0};id<${N_TRAIN_JOB};id++)); do
        job_id=$(aws batch submit-job \
            --job-name "${SCENARIO}_train_${AGENT}" \
            --job-queue experiment_workflow \
            --job-definition train_agent \
            --container-overrides command="""[\"python\", \"workflow/main.py\", \"step=train\", ${command_overwrite}, \"experiment.run_id='${id}'\"]""" \
            --output text --query jobId)

        job_id=$(aws batch submit-job \
            --job-name "${SCENARIO}_collect_${AGENT}_eval_traj" \
            --job-queue experiment_workflow \
            --job-definition collect_offline_data \
            --array-properties size=${EVAL_SIZE} \
            --depends-on jobId=${job_id} \
            --container-overrides command="""[\"python\", \"workflow/main.py\", \"step=eval\", ${command_overwrite}, \"experiment.run_id='${id}'\"]""" \
            --output text --query jobId)

        train_job_ids+=(${job_id})
        echo "job_id for run ${id}: ${job_id}"
    done


    AGENT_NAME="NeuralBoltzmann${N_FEATURE}f_${N_TRAIN_SIZE}"
    EVALUATE_AGENTS="[\"${AGENT_NAME}\"]"
    depends_on=" --depends-on "
    for id in "${train_job_ids[@]}"; do
        depends_on+="jobId=${id} "
    done

    job_id_4=$(aws batch submit-job \
        --job-name "${SCENARIO}_evaluate_agents_${AGENT_NAME}" \
        --job-queue experiment_workflow \
        --job-definition evaluate_agents \
        ${depends_on} \
        --container-overrides command="""['python', 'workflow/main.py', 'step=compute_metrics', 'synthetic_data=${SCENARIO}', 'pyenv=${ENV}', 'summary.agents=${EVALUATE_AGENTS}', 'summary.summary_file_prefix=${AGENT_NAME}_']""" \
        --output text --query jobId)

fi

#################################################################################################################################################
if ${RUN_DQN}; then
    echo "TRAIN DQN AGENTS"

    PARAM_OVERWRITE=(
        "agent.agent_params.optimizer.learning_rate=0.004643196981298455"
        "agent.agent_params.gamma=0.8792408738366871"
        "agent.agent_params.epsilon_greedy=0.15519727130455402"
        "agent.agent_params.q_network.dense_layer_1.units=16"
        "agent.agent_params.q_network.lstm_layer_1.units=4"
        "agent.agent_params.name=DQN${N_FEATURE}f_${N_TRAIN_SIZE}"
        "train.read_params_from_tuning_log=False"
        "train.train_data_size=${N_TRAIN_SIZE}"
    )
    AGENT='dqn'

    command_overwrite="""\"synthetic_data=${SCENARIO}\", \"pyenv=${ENV}\", \"agent=${AGENT}\", \"agent.agent_params.context_features=${FEATURES}\""""
    for param in "${PARAM_OVERWRITE[@]}"; do
        command_overwrite+=", \"${param}\""
    done
    echo ${command_overwrite}

    train_job_ids=()
    for ((id=${TRAIN_JOB_START_IDX=0};id<${N_TRAIN_JOB};id++)); do
        job_id=$(aws batch submit-job \
            --job-name "${SCENARIO}_train_${AGENT}" \
            --job-queue experiment_workflow \
            --job-definition train_agent \
            --container-overrides command="""[\"python\", \"workflow/main.py\", \"step=train\", ${command_overwrite}, \"experiment.run_id='${id}'\"]""" \
            --output text --query jobId)

        job_id=$(aws batch submit-job \
            --job-name "${SCENARIO}_collect_${AGENT}_eval_traj" \
            --job-queue experiment_workflow \
            --job-definition collect_offline_data \
            --array-properties size=${EVAL_SIZE} \
            --depends-on jobId=${job_id} \
            --container-overrides command="""[\"python\", \"workflow/main.py\", \"step=eval\", ${command_overwrite}, \"experiment.run_id='${id}'\"]""" \
            --output text --query jobId)

        train_job_ids+=(${job_id})
        echo "job_id for run ${id}: ${job_id}"
    done

    AGENT_NAME="DQN${N_FEATURE}f_${N_TRAIN_SIZE}"
    EVALUATE_AGENTS="[\"${AGENT_NAME}\"]"
    depends_on=" --depends-on "
    for id in "${train_job_ids[@]}"; do
        depends_on+="jobId=${id} "
    done

    job_id_4=$(aws batch submit-job \
        --job-name "${SCENARIO}_evaluate_agents_${AGENT_NAME}" \
        --job-queue experiment_workflow \
        --job-definition evaluate_agents \
        ${depends_on} \
        --container-overrides command="""['python', 'workflow/main.py', 'step=compute_metrics', 'synthetic_data=${SCENARIO}', 'pyenv=${ENV}', 'summary.agents=${EVALUATE_AGENTS}', 'summary.summary_file_prefix=${AGENT_NAME}_']""" \
        --output text --query jobId)
fi

################################################################################################################################################
if ${RUN_PPO}; then
    echo "TRAIN PPO AGENTS"

    PARAM_OVERWRITE=(
        "agent.agent_params.optimizer.learning_rate=0.006380757050779136"
        "agent.agent_params.importance_ratio_clipping=0.7849416213439981"
        "agent.agent_params.discount_factor=0.8198518927547219"
        "agent.agent_params.entropy_regularization=0.1012597119847905"
        "agent.agent_params.num_epochs=12"
        "agent.agent_params.actor_net.fc_layer_params=[32,8]"
        "agent.agent_params.value_net.fc_layer_params=[32,8]"
        "agent.agent_params.name=PPO${N_FEATURE}f_${N_TRAIN_SIZE}"
        "train.read_params_from_tuning_log=False"
        "train.train_data_size=${N_TRAIN_SIZE}"
    )
    AGENT='ppo'

    command_overwrite="""\"synthetic_data=${SCENARIO}\", \"pyenv=${ENV}\", \"agent=${AGENT}\", \"agent.agent_params.context_features=${FEATURES}\""""
    for param in "${PARAM_OVERWRITE[@]}"; do
        command_overwrite+=", \"${param}\""
    done
    echo ${command_overwrite}

    train_job_ids=()
    for ((id=${TRAIN_JOB_START_IDX=0};id<${N_TRAIN_JOB};id++)); do
        job_id=$(aws batch submit-job \
            --job-name "${SCENARIO}_train_${AGENT}" \
            --job-queue experiment_workflow \
            --job-definition train_agent \
            --container-overrides command="""[\"python\", \"workflow/main.py\", \"step=train\", ${command_overwrite}, \"experiment.run_id='${id}'\"]""" \
            --output text --query jobId)

        job_id=$(aws batch submit-job \
            --job-name "${SCENARIO}_collect_${AGENT}_eval_traj" \
            --job-queue experiment_workflow \
            --job-definition collect_offline_data \
            --array-properties size=${EVAL_SIZE} \
            --depends-on jobId=${job_id} \
            --container-overrides command="""[\"python\", \"workflow/main.py\", \"step=eval\", ${command_overwrite}, \"experiment.run_id='${id}'\"]""" \
            --output text --query jobId)

        train_job_ids+=(${job_id})
        echo "job_id for run ${id}: ${job_id}"
    done

    AGENT_NAME="PPO${N_FEATURE}f_${N_TRAIN_SIZE}"
    EVALUATE_AGENTS="[\"${AGENT_NAME}\"]"
    depends_on=" --depends-on "
    for id in "${train_job_ids[@]}"; do
        depends_on+="jobId=${id} "
    done

    job_id_4=$(aws batch submit-job \
        --job-name "${SCENARIO}_evaluate_agents_${AGENT_NAME}" \
        --job-queue experiment_workflow \
        --job-definition evaluate_agents \
        ${depends_on} \
        --container-overrides command="""['python', 'workflow/main.py', 'step=compute_metrics', 'synthetic_data=${SCENARIO}', 'pyenv=${ENV}', 'summary.agents=${EVALUATE_AGENTS}', 'summary.summary_file_prefix=${AGENT_NAME}_']""" \
        --output text --query jobId)
fi