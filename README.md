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

### Data Precessing

See preprocess/XXXDataset.ipynb for details


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

Then train retention model:

```
bash train_retention_env.sh
```

## 1.2. Simulated Environment

For example of using the simulator:

> RetentionSimulator.ipynb


# 2. Benchmarks

## 2.1 Listwise Recommendation

### 2.1.1 Setup

Evaluation metrics and protocol

### 2.1.2 Training

```
bash train_td3.sh
```

### 2.1.3 Baselines

Results

## 2.2 Whole-session Recommendation

Whole-session user interaction involves multiple request-feedback loop.

### 2.2.1 Setup

Evaluation metrics and protocol

### 2.2.2 Training

```
bash train_td3.sh
```

### 2.2.3 Baselines

Results

## 2.3 Retention Optimization

User retention happens after leaving of previous session and identifies the beginning of the next session.

### 2.3.1 Setup

Evaluation metrics and protocol

### 2.3.2 Training

```
bash train_td3.sh
```

### 2.3.3 Baselines

Results


# 3. Result Observation

Training curves check:

> TrainingObservation.ipynb
