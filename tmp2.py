import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# ==== 1. MFCC 추출 함수 ====
def extract_mfcc(path, n_mfcc=40):
    y, sr = librosa.load(path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc.T  # (frame, n_mfcc)

# ==== 2. 데이터 로딩 ====
x = []   # 특징 벡터 모음
y = []   # 라벨 (0: 남성, 1: 여성)

for i in range(1, 6):  # m1.wav ~ m5.wav, f1.wav ~ f5.wav
    mfcc_m = extract_mfcc(f'./data/m{i}.wav')  # 남성
    mfcc_f = extract_mfcc(f'./data/f{i}.wav')  # 여성

    x.append(mfcc_m)
    y.extend([0] * len(mfcc_m))

    x.append(mfcc_f)
    y.extend([1] * len(mfcc_f))

x = np.vstack(x)  # (총 frame 수, n_mfcc)
y = np.array(y)

# ==== 3. 특성 정규화 ====
scaler = StandardScaler()
x = scaler.fit_transform(x)

# ==== 4. 교차검증 및 GMM 분류 ====
skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
accuracies = []

for fold, (train_idx, test_idx) in enumerate(skf.split(x, y), 1):
    X_train, X_test = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # 클래스별로 GMM 모델 생성
    gmm_male = GaussianMixture(n_components=4, covariance_type='diag', random_state=0)
    gmm_female = GaussianMixture(n_components=4, covariance_type='diag', random_state=0)

    gmm_male.fit(X_train[y_train == 0])
    gmm_female.fit(X_train[y_train == 1])

    # 테스트 데이터에 대해 log-likelihood 계산
    score_m = gmm_male.score_samples(X_test)
    score_f = gmm_female.score_samples(X_test)

    # 남성 모델 점수가 높으면 0, 아니면 1
    y_pred = (score_f > score_m).astype(int)
    acc = np.mean(y_pred == y_test) * 100
    accuracies.append(acc)

    print(f"Fold {fold} Accuracy: {acc:.2f}%")

# ==== 5. 평균 및 표준편차 출력 ====
mean_acc = np.mean(accuracies)
std_acc = np.std(accuracies)

print(f"\nAverage Accuracy: {mean_acc:.2f}%")
print(f"Standard Deviation: {std_acc:.2f}%")