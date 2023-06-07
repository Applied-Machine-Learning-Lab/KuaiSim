mkdir -p output


mkdir -p output/Kuairand_Pure/

output_path="output/Kuairand_Pure"

mkdir ${output_path}/agents

log_name="user_KRMBUserResponseWithBias_lr0.0003_reg0_nlayer2"

# environment arguments
ENV_CLASS='KREnvironment_SlateRec'
SLATE_SIZE=6
MAX_STEP=5
EP_BS=64
RHO=0.2

# policy arguments
POLICY_CLASS='PointwiseRanker'
POS_OFFSET=0.4
NEG_OFFSET=0.1
LOSS='Reward BCE'

# buffer arguments
BUFFER_CLASS='BaseBuffer'
BUFFER_SIZE=100000

# agent arguments
AGENT_CLASS='ListRecOnlineAgent'
REWARD_FUNC='get_immediate_reward'
N_ITER=10000
START_STEP=100
INITEP=0.01
ELBOW=0.1
EXPLORE_RATE=1.0
BS=128


for REG in 0 # 0.001 0.0001
do
    for ACTOR_LR in 0.0001 # 0.0003
    do
        for POS_OFFSET in 0.4
        do
            for SEED in 11 # 13 17 19 23
            do
                file_key=${AGENT_CLASS}_${POLICY_CLASS}_actor${ACTOR_LR}_pos${POS_OFFSET}_neg${NEG_OFFSET}_niter${N_ITER}_reg${REG}_ep${INITEP}_bs${BS}_epbs${EP_BS}_seed${SEED}

                mkdir -p ${output_path}/agents/${file_key}/

                CUDA_VISIBLE_DEVICES=1 python train_online_policy.py\
                    --env_class ${ENV_CLASS}\
                    --policy_class ${POLICY_CLASS}\
                    --critic_class ${CRITIC_CLASS}\
                    --buffer_class ${BUFFER_CLASS}\
                    --agent_class ${AGENT_CLASS}\
                    --seed ${SEED}\
                    --cuda 0\
                    --max_step_per_episode ${MAX_STEP}\
                    --initial_temper ${MAX_STEP}\
                    --uirm_log_path ${output_path}/env/log/${log_name}.model.log\
                    --slate_size ${SLATE_SIZE}\
                    --episode_batch_size ${EP_BS}\
                    --item_correlation ${RHO}\
                    --policy_action_hidden 256 64\
                    --ptranker_pos_offset ${POS_OFFSET}\
                    --ptranker_neg_offset ${NEG_OFFSET}\
                    --loss ${LOSS}\
                    --state_user_latent_dim 16\
                    --state_item_latent_dim 16\
                    --state_transformer_enc_dim 32\
                    --state_transformer_n_head 4\
                    --state_transformer_d_forward 64\
                    --state_transformer_n_layer 3\
                    --state_dropout_rate 0.1\
                    --buffer_size ${BUFFER_SIZE}\
                    --reward_func ${REWARD_FUNC}\
                    --n_iter ${N_ITER}\
                    --train_every_n_step 1\
                    --start_policy_train_at_step ${START_STEP}\
                    --initial_epsilon ${INITEP}\
                    --final_epsilon ${INITEP}\
                    --elbow_epsilon ${ELBOW}\
                    --explore_rate ${EXPLORE_RATE}\
                    --check_episode 10\
                    --save_episode 200\
                    --test_episode 200\
                    --save_path ${output_path}/agents/${file_key}/model\
                    --batch_size ${BS}\
                    --actor_lr ${ACTOR_LR}\
                    --actor_decay ${REG}\
                    > ${output_path}/agents/${file_key}/log
            done
        done
    done
done
