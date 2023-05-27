
output_path='output/Kuairand_Pure'
# pick your 
KRMB_MODEL_KEY='user_KRMBUserResponse_lr0.0001_reg0.01_nlayer2'

python generate_session_data.py\
    --behavior_model_log_file ${output_path}/env/log/${KRMB_MODEL_KEY}.model.log\
    --data_output_path dataset/Kuairand-Pure/cross_session_${KRMB_MODEL_KEY}.csv
