# KRLBenchmark

Online simulator for RL-based recommendation

# 0.Setup

```
conda create -n KRL python=3.8
conda activate KRL
conda install pytorch torchvision -c pytorch
conda install pandas matplotlib scikit-learn tqdm ipykernel
python -m ipykernel install --user --name KRL --display-name "KRL"
```

# 1. Simulator Setup

### Data Processing

See preprocess/KuaiRandDataset.ipynb for details


## 1.1 User Model

1.1 Immediate User Response Model

Example raw data format in preprocessed KuaiRand: 

> (session_id, request_id, user_id, video_id, date, time, is_click, is_like, is_comment, is_forward, is_follow, is_hate, long_view)

Example item meta data format in preprocessed KuaiRand: 

> (video_id, video_type, upload_type, music_type, log_duration, tag)

Example user meta data format in preprocessed KuaiRand: 

> (user_active_degree, is_live_streamer, is_video_author, follow_user_num_range, fans_user_num_range, friend_user_num_range, register_days_range, onehot_feat{0,1,6,9,10,11,12,13,14,15,16,17})

```
bash train_multi_behavior_user_response.sh
```

Note: multi-behavior user response models consists the state_encoder that is assumed to be the ground truth user state transition model.

1.2 User Retention Model

Pick a multi-behavior user response model for cross-session generation and retention model training, change the shell script accordingly (by setting the keyword 'KRMB_MODEL_KEY').

Generate user retention data in format:

> (session_id, user_id, session_enc, return_day)

```
bash generate_session_data.sh
```

# 2. Benchmarks

## 2.1 Listwise Recommendation

### 2.1.1 Setup

Evaluation metrics and protocol

**List-wise reward (L-reward)** is the average of item-wise immediate reward. We use both the average L-reward and the max L-reward across user requests in a mini-batch. 

**Reward-based NDCG (R-NDCG)** generalizes the standard NDCG metric where the item-wise reward becomes the relevance label, and the IDCG is agnostic to the model being evaluated.
**Reward-weighted mean reciprocal rank(R-MRR)** generalizes the standard MRR metric but replaces the item label with the item-wise reward. For both metrics, a larger value means that the learned policy performs better on the offline data. 

**Coverage** describes the number of distinct items exposed in a mini-batch. 

**Intra-list diversity (ILD)** estimates the embedding-based dissimilarity between items in each recommended list

### 2.1.2 Training

```
bash train_{model name}_krpure_requestlevel.sh
```

### 2.1.3 Baselines

| Algorithm | Average L-reward | Max L-reward |  Coverage   |    ILD    |
| :-------: | :--------------: | :----------: | :---------: | :-------: |
|    CF     |    **2.253**     |    4.039     |   100.969   |   0.543   |
| ListCVAE  |      2.075       |  **4.042**   | **446.100** | **0.565** |
|    PRM    |      2.174       |    3.811     |   27.520    |   0.53    |



## 2.2 Whole-session Recommendation

Whole-session user interaction involves multiple request-feedback loops.

### 2.2.1 Setup

Evaluation metrics and protocol

**Whole-session reward**: total reward is the average sum of immediate rewards for each session. The average reward is the average of total reward for each request.

**Depth** represents how many interactions before the user leaves.

### 2.2.2 Training

```
bash train_{model name}_krpure_wholesession.sh
```

### 2.2.3 Baselines

| Algorithm |   Depth   | Average reward | Total reward | Coverage  |    ILD     |
| :-------: | :-------: | :------------: | :----------: | :-------: | :--------: |
|    TD3    |   14.63   |     0.6476     |    9.4326    |   24.20   |   0.9864   |
|    A2C    |   14.02   |     0.5950     |    8.3905    |   27.41   |   0.9870   |
|   DDPG    |   14.89   |     0.6841     |   10.0873    |   20.95   |   0.9850   |
|    HAC    | **14.98** |   **0.6895**   | **10.1742**  | **35.70** | **0.9874** |



## 2.3 Retention Optimization

User retention happens after leaving of previous session and identifies the beginning of the next session.

### 2.3.1 Setup

Evaluation metrics and protocol

**Return time** is the average time gap between the last request of session and the first request of session. 

**User retention** is the average ratio of visiting the system again.

### 2.3.2 Training

```
bash train_{model name}_krpure_crosssession.sh
```

### 2.3.3 Baselines

| Algorithm | Return time **↓** | User retention ↑ |
| :-------: | :---------------: | :--------------: |
|    CEM    |       3.573       |      0.572       |
|    TD3    |       3.556       |      0.581       |
|   RLUR    |     **3.481**     |     **0.607**     |




# 3. Result Observation

Training curves check:

> TrainingObservation.ipynb

# 4. Other Analysis Experiments

Training other simulators:

> bash train_ddpg_krpure_wholesession_{simulator name}.sh

Training on ML-1m dataset:

> bash train_ddpg_krpure_wholesession_ml.sh


# Reference

[1] Zhao, K., Liu, S., Cai, Q., Zhao, X., Liu, Z., Zheng, D., ... & Gai, K. (2023). KuaiSim: A comprehensive simulator for recommender systems. arXiv preprint arXiv:2309.12645.

# BibTeX entry

Please cite the paper if you use this code in your work:


```
@article{zhao2023kuaisim,
  title={KuaiSim: A comprehensive simulator for recommender systems},
  author={Zhao, Kesen and Liu, Shuchang and Cai, Qingpeng and Zhao, Xiangyu and Liu, Ziru and Zheng, Dong and Jiang, Peng and Gai, Kun},
  journal={arXiv preprint arXiv:2309.12645},
  year={2023}
}
```
