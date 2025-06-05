import librosa
#import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.mixture import GaussianMixture

def MFCC(data):
    y ,sr = librosa.load(data,sr=16000)
    mfcc = librosa.feature.mfcc(y=y,sr=sr,n_mfcc=40,hop_length = 128)
    return mfcc.T

"""
for i in range(1,6):
    male_mfcc = MFCC(f'./data/m{i}.wav')
    female_mfcc = MFCC(f'./data/f{i}.wav')
    plt.subplot(2, 5, i)
    plt.title(f'Male {i}')
    librosa.display.specshow(male_mfcc)
    plt.subplot(2, 5, i+5)
    plt.title(f'FeMale {i}')
    librosa.display.specshow(female_mfcc)
"""
x = []
y = []
# path "./data/{  }"
for i in range(1,6):
    x.append(MFCC(f'./data/m{i}.wav'))
    y.extend([0]*len(MFCC(f'./data/m{i}.wav')))#male
    x.append(MFCC(f'./data/f{i}.wav'))
    y.extend([1]*len(MFCC(f'./data/f{i}.wav')))#female

x = np.vstack(x) # to array
y = np.array(y)

#print(x.shape)
#print(y.shape)

skf = StratifiedKFold(n_splits=4,shuffle=True)

accurate = []

for i, (train_i, test_i) in enumerate(skf.split(x, y)):
    x_train, x_test = x[train_i], x[test_i]
    y_train, y_test = y[train_i], y[test_i]

    gmm_male = GaussianMixture(n_components=4, covariance_type='full', random_state=5)
    gmm_female = GaussianMixture(n_components=4, covariance_type='full', random_state=5)

    gmm_male.fit(x_train[y_train == 0])
    gmm_female.fit(x_train[y_train == 1])

    score_m = gmm_male.score_samples(x_test)
    score_f = gmm_female.score_samples(x_test)

    y_pred = (score_f > score_m).astype(int)
    acc = np.mean(y_pred == y_test) * 100
    accurate.append(acc)

    print(f"{i+1} acc: {acc:.2f}")

print(f"Mean Accuracy:{np.mean(accurate):.2f}, Standard Deviation: {np.std(accurate):.2f}")

#plt.show()