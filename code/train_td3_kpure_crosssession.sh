mkdir -p output

# RL4RS environment

mkdir -p output/Kuairand_Pure/
mkdir -p output/Kuairand_Pure/agents/

env_path="output/Kuairand_Pure/env/"
output_path="output/Kuairand_Pure/agents"

log_name="user_KRMBUserResponse_lr0.0001_reg0_nlayer2"

# environment arguments
ENV_CLASS='KRCrossSessionEnvironment_GPU'
MAX_SESSION=6
MAX_RET_DAY=10
MAX_STEP=5
RET_DAY_BIAS=0.4
FEEDBACK_INF_RETURN=0.1
SLATE_SIZE=6
EP_BS=32
RHO=0.2

# policy arguments
POLICY_CLASS='ActionTransformer'

# critic arguments
CRITIC_CLASS='QCritic'

# buffer arguments
BUFFER_CLASS='CrossSessionBuffer'
BUFFER_SIZE=100000

# agent arguments
AGENT_CLASS='CrossSessionTD3'
GAMMA=0.9
REWARD_FUNC='get_retention_reward'
N_ITER=30000
START_STEP=100
INITEP=0.01
EXPLORE_RATE=1.0
ELBOW=0.1
BS=128
NOISE=0.1


for NOISE in 0.1
do
    for REG in 0.00001 # 0.001 0.0001
    do
        for CRITIC_LR in 0.001 # 0.0001
        do
            for ACTOR_LR in 0.0001 # 0.0003 # 0.00001
            do
                for SEED in 11 # 11 13 17 19 23
                do
                    file_key=td3_${POLICY_CLASS}_${ENV_CLASS}_actor${ACTOR_LR}_critic${CRITIC_LR}_niter${N_ITER}_reg${REG}_ep${INITEP}_noise${NOISE}_bs${BS}_epbs${EP_BS}_fret${FEEDBACK_INF_RETURN}_seed${SEED}
                    
                    mkdir -p ${output_path}/${file_key}/

                    python train_td3.py\
                        --seed ${SEED}\
                        --cuda 0\
                        --env_class ${ENV_CLASS}\
                        --uirm_log_path ${env_path}log/${log_name}.model.log\
                        --max_n_session ${MAX_SESSION}\
                        --max_return_day ${MAX_RET_DAY}\
                        --initial_temper ${MAX_STEP}\
                        --next_day_return_bias ${RET_DAY_BIAS}\
                        --feedback_influence_on_return ${FEEDBACK_INF_RETURN}\
                        --slate_size ${SLATE_SIZE}\
                        --episode_batch_size ${EP_BS}\
                        --item_correlation ${RHO}\
                        --max_step_per_episode ${MAX_STEP}\
                        --policy_class ${POLICY_CLASS}\
                        --policy_user_latent_dim 16\
                        --policy_item_latent_dim 16\
                        --policy_enc_dim 32\
                        --policy_attn_n_head 4\
                        --policy_transformer_d_forward 64\
                        --policy_transformer_n_layer 2\
                        --policy_hidden_dims 128\
                        --policy_dropout_rate 0.1\
                        --critic_class ${CRITIC_CLASS}\
                        --critic_hidden_dims 128 32\
                        --critic_dropout_rate 0.1\
                        --buffer_class ${BUFFER_CLASS}\
                        --buffer_size ${BUFFER_SIZE}\
                        --agent_class ${AGENT_CLASS}\
                        --actor_lr ${ACTOR_LR}\
                        --critic_lr ${CRITIC_LR}\
                        --actor_decay ${REG}\
                        --critic_decay ${REG}\
                        --target_mitigate_coef 0.01\
                        --gamma ${GAMMA}\
                        --reward_func ${REWARD_FUNC}\
                        --n_iter ${N_ITER}\
                        --episode_batch_size ${EP_BS}\
                        --batch_size ${BS}\
                        --train_every_n_step 1\
                        --start_policy_train_at_step ${START_STEP}\
                        --initial_epsilon ${INITEP}\
                        --final_epsilon ${INITEP}\
                        --elbow_epsilon ${ELBOW}\
                        --explore_rate ${EXPLORE_RATE}\
                        --check_episode 10\
                        --save_episode 200\
                        --save_path ${output_path}/${file_key}/model\
                        --batch_size ${BS}\
                        --noise_var ${NOISE}\
                        --noise_clip 1.0\
                        > ${output_path}/${file_key}/log
                done
            done
        done
    done
done