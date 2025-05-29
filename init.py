import os
import numpy as np
import librosa
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score

def MFCC(file_path, n_mfcc=5):
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc.T

def load_files(file_names, data_dir):
    features = []
    for name in file_names:
        path = os.path.join(data_dir, f'{name}.wav')
        mfcc = MFCC(path)
        features.append(mfcc)
    return np.vstack(features)

def predict(file_path, male_gmm, female_gmm):
    mfcc = MFCC(file_path)
    male_score = male_gmm.score(mfcc)
    female_score = female_gmm.score(mfcc)
    return 'male' if male_score > female_score else 'female'

# ====== 데이터 정의 ======
data_dir = './data'

# 학습용 데이터 (m1~m4, f1~f4)
train_male = ['m1', 'm2', 'm3', 'm4']
train_female = ['f1', 'f2', 'f3', 'f4']

# 테스트용 데이터 (m5, f5)
test_files = ['m5', 'f5']
test_labels = ['male', 'female']

# ====== 모델 학습 ======
male_features = load_files(train_male, data_dir)
female_features = load_files(train_female, data_dir)

male_gmm = GaussianMixture(n_components=6, covariance_type='diag', random_state=0)
female_gmm = GaussianMixture(n_components=6, covariance_type='diag', random_state=0)

male_gmm.fit(male_features)
female_gmm.fit(female_features)

# ====== 모델 평가 ======
predictions = []
for fname in test_files:
    path = os.path.join(data_dir, f'{fname}.wav')
    pred = predict(path, male_gmm, female_gmm)
    predictions.append(pred)
    print(f"{fname}.wav → 예측: {pred}")

# 정확도 계산
acc = accuracy_score(test_labels, predictions)
print(f"\n정확도: {acc * 100:.2f}%")