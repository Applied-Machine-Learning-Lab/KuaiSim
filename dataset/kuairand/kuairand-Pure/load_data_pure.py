import pandas as pd


# The "nrows=10000" argument reads the first 10000 lines of each file

df_rand = pd.read_csv("data/log_random_4_22_to_5_08_pure.csv", nrows=10000)

df1 = pd.read_csv("data/log_standard_4_08_to_4_21_pure.csv", nrows=10000)
df2 = pd.read_csv("data/log_standard_4_22_to_5_08_pure.csv", nrows=10000)

user_features = pd.read_csv("data/user_features_pure.csv")

video_features_basic = pd.read_csv("data/video_features_basic_pure.csv", nrows=10000)
video_features_statistics = pd.read_csv("data/video_features_statistic_pure.csv", nrows=10000)

print("===================================================")
print("The random data in 'log_random_4_22_to_5_08_pure.csv'")
print("---------------------------------------------------")
print(df_rand)


print("===================================================")
print("The standard data in 'log_standard_XXX.csv'")
print("---------------------------------------------------")
print(df1)

print("===================================================")
print("The user features in 'user_features_pure.csv'")
print("---------------------------------------------------")
print(user_features)

print("===================================================")
print("The basic video features in 'video_features_basic.csv'")
print("---------------------------------------------------")
print(video_features_basic)

print("===================================================")
print("The statistical features of videos in 'video_features_statistic_pure.csv'")
print("---------------------------------------------------")
print(video_features_statistics)

a = 1