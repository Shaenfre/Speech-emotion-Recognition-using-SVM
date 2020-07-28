import librosa
import soundfile
import os, glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from scipy.stats import moment
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
def extract_ftrs(file_name):
    x, sample_rate = librosa.load(file_name, sr=None)
    result = np.array([])
    mfcc = librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    ddelta = librosa.feature.delta(mfcc, order=2)
    #calculating mean part
    mfcc_m = np.mean(mfcc.T,axis=0)
    result = np.hstack((result,mfcc_m))
    deltm = np.mean(delta.T,axis=0)
    result = np.hstack((result, deltm))
    delt2m = np.mean(ddelta.T, axis=0)
    result = np.hstack((result, delt2m))
    #calculating standard deviation part
    mfcc_s = np.std(mfcc.T, axis=0)
    result = np.hstack((result,mfcc_s))
    delt_s = np.std(delta.T, axis=0)
    result = np.hstack((result, delt_s))
    ddelta_s = np.std(ddelta.T, axis=0)
    result = np.hstack((result, ddelta_s))
    #calculating 3rd moments
    mfcc_3 = moment(mfcc.T, moment = 3, axis=0)
    result = np.hstack((result, mfcc_3))
    delt_3 = moment(delta.T, moment = 3, axis=0)
    result = np.hstack((result, delt_3))
    ddelta_3 = moment(ddelta.T, moment = 3, axis=0)
    result = np.hstack((result, ddelta_3))
    #calcualting 4th moment
    mfcc_4 = moment(mfcc.T, moment=4, axis=0)
    result = np.hstack((result, mfcc_4))
    delt_4 = moment(delta.T, moment=4, axis=0)
    result = np.hstack((result, delt_4))
    ddelta_4 = moment(ddelta.T, moment=4, axis=0)
    result = np.hstack((result, ddelta_4))
    #calculating 5th moment
    mfcc_5 = moment(mfcc.T, moment=5, axis=0)
    result = np.hstack((result, mfcc_5))
    delt_5 = moment(delta.T, moment=5, axis=0)
    result = np.hstack((result, delt_5))
    ddelta_5 = moment(ddelta.T, moment=5, axis=0)
    result = np.hstack((result, ddelta_5))

    return result
    # with soundfile.SoundFile(file_name) as sound_file:
    #     X = sound_file.read(dtype = 'float32')
    #     sample_rate = sound_file.samplerate
    #     result = []
    #     mfccs = librosa.feature.mfcc(y = X, sr = sample_rate, n_mfcc = 13)
    #     result.append(mfccs)
    #     mfcc_delta = librosa.feature.delta(mfccs)
    #     result.append(result, mfcc_delta)
    #     mfcc_delta2 = librosa.feature.delta(mfcc, order = 2)
    #     result.append(result, mfcc_delta2)
    # return result
# emotions = {
#     'a' : 'angry',
#     'd' : 'disgust',
#     'f' : 'fear',
#     'h' : 'happiness',
#     'n' : 'neutral',
#     'sa' : 'sad',
#     'su' : 'suprised'
# }
def load_data():
    x, y = [], []
    print(os.getcwd())
    os.chdir(r"/Users/shubham/Desktop/Audio_data")
    for file_name in glob.glob("*.wav"):
        file_name = os.path.basename(file_name)
        #print(file_name)
        emotion = file_name[0]
        feature = extract_ftrs(file_name)
        #print(type(feature))
        x.append(feature)
        #print(np.shape(x))
        y.append(emotion)
        X = np.array(x)
        #print(np.shape(X))
        Y = np.array(y)
    return X,Y
x,y=load_data()
print(len(x))
print(len(y))
print(np.shape(x))
print('done')
X_train, X_test, y_train, y_test=train_test_split(x, y, test_size=0.1, stratify=y)
print('done')
print(np.shape(X_train))
Scaler = MinMaxScaler(feature_range=(0,1))
Scaler.fit(X_train)
X_train_scld = Scaler.transform(X_train)
X_test_scld = Scaler.transform(X_test)
le = LabelEncoder()
le.fit(y_train)
y_train_enc = le.transform(y_train)
y_test_enc = le.transform(y_test)
# y_train_enc.reshape(1, -1)
# y_test_enc.reshape(1, -1)
# ohe = OneHotEncoder()
# y_train_enco = ohe.fit_transform(y_train_enc)
# y_test_enco = ohe.fit_transform(y_test_enc)
# model = SVC(kernel='linear')
# model.fit(X_train_scld, y_train_enc)
# print(model.score(X_test_scld, y_test_enc))
#print(cross_val_score(SVC(gamma = 'auto',C=10,kernel='linear'),X_train,y_train,cv=5).mean())
rs = GridSearchCV(SVC(),{
    'C':[100,500],
    'gamma':[0.001,0.1,1],
    'kernel':['linear','poly','rbf']
    },
      cv=5,
      return_train_score=False,
      )
rs.fit(X_train_scld,y_train_enc)
data = pd.DataFrame(rs.cv_results_)[['param_C','param_gamma','param_kernel', 'mean_test_score']]
print(data)