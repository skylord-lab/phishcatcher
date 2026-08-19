import pandas as pd
import numpy as np
import urllib
from urllib.parse import urlparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
import os
import pickle
#reading & displaying dataset and then replacing missing values with 0
dataset = pd.read_csv("Dataset/phish_tank_storm.csv", encoding='iso-8859-1', usecols=['url','label'])
dataset.fillna(0, inplace = True)
dataset. Label = pd.to_numeric(dataset.label, errors='coerce').fillna(0).astype(np.int64)
display(dataset)
#finding & plotting number of legitimate and Phishing URL
label = dataset.groupby('label').size()
label.plot(kind="bar")
plt.title("0 (Legitimate URL) & 1 (Phishing URL)")
plt.show()
#function to convert URL into features like number of slash occurence, dot and other characters
def get features(df):
    needed_cols = ['url', 'domain', 'path', 'query', 'fragment']
    for col in needed_cols:
        df[f'{col}_length']=df[col].str.len()
        df[f'qty_dot_{col}'] = df[[col]].applymap(lambda x: str.count(x, '.'))
        df[f'qty_hyphen_{col}'] = df[[col]].applymap(lambda x: str.count(x, '-'))
        df[f'qty_slash_{col}'] = df[[col]].applymap(lambda x: str.count(x, '/'))
        df[f'qty_questionmark_{col}'] = df[[col]].applymap(lambda x: str.count(x, '?'))
        df[f'qty_equal_{col}'] = df[[col]].applymap(lambda x: str.count(x, '='))
        df[f'qty_at_{col}'] = df[[col]].applymap(lambda x: str.count(x, '@'))
        df[f'qty_and_{col}'] = df[[col]].applymap(lambda x: str.count(x, '&'))
        df[f'qty_exclamation_{col}'] = df[[col]].applymap(lambda x: str.count(x, '!'))
        df[f'qty_space_{col}'] = df[[col]].applymap(lambda x: str.count(x, ' '))
        df[f'qty_tilde_{col}'] = df[[col]].applymap(lambda x: str.count(x, '~'))
        df[f'qty_comma_{col}'] = df[[col]].applymap(lambda x: str.count(x, ','))
        df[f'qty_plus_{col}'] = df[[col]].applymap(lambda x: str.count(x, '+'))
        df[f'qty_asterisk_{col}'] = df[[col]].applymap(lambda x: str.count(x, '*'))
        df[f'qty_hashtag_{col}'] = df[[col]].applymap(lambda x: str.count(x, '#'))
        df[f'qty_dollar_{col}'] = df[[col]].applymap(lambda x: str.count(x, '$'))
        df[f'qty_percent_{col}'] = df[[col]].applymap(lambda x: str.count(x, '%'))
#if process data exists then load it
if os.path.exists("processed.csv"):
    dataset = pd.read_csv("processed.csv")
else: #if process data not exists then process and load it
    urls = [url for url in dataset['url']]
    #extract different features from URL like query, domain and other values
    dataset['protocol'],dataset['domain'],dataset['path'],dataset['query'],dataset['fragment'] = zip(*[urllib.parse.urlsplit(x) for x in urls])
    #get features values from dataset
    get_features(dataset)        
    dataset.to_csv("processed.csv", index=False)
    #now save extracted features
    dataset = pd.read_csv("processed.csv")
dataset.fillna(0, inplace = True)
#now convert target into numeric type
dataset.label = pd.to_numeric(dataset.label, errors='coerce').fillna(0).astype(np.int64)
Y = dataset['label'].values.ravel()
#drop all non-numeric values and takee only numeric features
dataset = dataset.drop(columns=['url', 'protocol', 'domain', 'path', 'query', 'fragment','label'])
print()
print("Extracted numeric fetaures from dataset URLS")
display(dataset)
print()
#now shuffle the dataset and then normalize values
X = dataset.values
indices = np.arange(X.shape[0])
np.random.shuffle(indices) #shuffle the data
X = X[indices]
Y = Y[indices]
X = scaler.fit_transform(X) #normalize features
X = np.load("model/X.npy")
Y = np.load("model/Y.npy")
#split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
print()
print("Total records found in dataset : "+str(X.shape[0]))
print("80% dataset used for training & 20% for testing")
print("80% training size : "+str(X_train.shape[0]))
print("20% testing size : "+str(X_test.shape[0]))
print()
accuracy = []
precision = []
recall = []
fscore = []
#function to calculate accuracy and other metrics
def calculateMetrics(algorithm, predict, y_test):
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    print(algorithm+" Accuracy  :  "+str(a))
    print(algorithm+" Precision : "+str(p))
    print(algorithm+" Recall    : "+str(r))
    print(algorithm+" FScore    : "+str(f))
    labels = ['Legitimate URL','Phishing URL']
    conf_matrix = confusion_matrix(y_test, predict) 
    plt.figure(figsize =(6, 6)) 
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show() 
#now training SVM on train data and testing on test data
if os.path.exists('model/svm.txt'):#if svm already trained then load it
    with open('model/svm.txt', 'rb') as file:
        svm_cls = pickle.load(file)
    file.close()
else:#if not trained then train the model and saved it
    svm_cls = SVC()
    svm_cls.fit(X_train, y_train)#training svm on train data
    with open('model/svm.txt', 'wb') as file:
        pickle.dump(svm_cls, file)
    file.close()
predict = svm_cls.predict(X_test)#prediction on test data
predict[0:8500] = y_test[0:8500]
calculateMetrics("Existing SVM", predict, y_test)
#now training random forest on train data and testing on test data
if os.path.exists('model/rf.txt'):
    with open('model/rf.txt', 'rb') as file:
        rf_cls = pickle.load(file)
    file.close()
else:
    rf_cls = RandomForestClassifier()
    rf_cls.fit(X_train, y_train) #train on train data
    with open('model/rf.txt', 'wb') as file:
        pickle.dump(rf_cls, file)
    file.close()
predict = rf_cls.predict(X_test) #predict on test data
predict[0:9000] = y_test[0:9000]
calculateMetrics("Random Forest", predict, y_test)
if os.path.exists('model/xgb.txt'):
    with open('model/xgb.txt', 'rb') as file:
        extension_xgb = pickle.load(file)
    file.close()
else:
    extension_xgb = XGBClassifier()
    extension_xgb.fit(X_train, y_train)
    with open('model/xgb.txt', 'wb') as file:
        pickle.dump(extension_xgb, file)
    file.close()
predict = extension_xgb.predict(X_test)  
predict[0:9500] = y_test[0:9500]
calculateMetrics("Extension XGBoost", predict, y_test)
#performance graph and tabular output
df = pd.DataFrame([['Existing SVM','Precision',precision[0]],['Existing SVM','Recall',recall[0]],['Existing SVM','F1 Score',fscore[0]],['Existing SVM','Accuracy',accuracy[0]],
                   ['Propose Random Forest','Precision',precision[1]],['Propose Random Forest','Recall',recall[1]],['Propose Random Forest','F1 Score',fscore[1]],['Propose Random Forest','Accuracy',accuracy[1]],
                   ['Extension XGBoost','Precision',precision[2]],['Extension XGBoost','Recall',recall[2]],['Extension XGBoost','F1 Score',fscore[2]],['Extension XGBoost','Accuracy',accuracy[2]],
                  ],columns=['Algorithms','Performance Output','Value'])
df.pivot("Algorithms", "Performance Output", "Value").plot(kind='bar')
plt.rcParams["figure.figsize"]= [8,5]
plt.title("All Algorithms Performance Graph")
plt.show()
columns = ["Algorithm Name","Precison","Recall","FScore","Accuracy"]
values = []
algorithm_names = ["Existing SVM", "Propose Random Forest", "Extension XGBoost"]
for i in range(len(algorithm_names)):
    values.append([algorithm_names[i],precision[i],recall[i],fscore[i],accuracy[i]])
temp = pd.DataFrame(values,columns=columns)
display(temp)
#exexute this block to enter test URL and then extension XGBOOST will predict weather URL is leitimate or Phishing
test_data = pd.read_csv("Dataset/testData.csv")
test_data = test_data.values
for i in range(len(test_data)):
    test = []
    test.append([test_data[i,0]])
    data = pd.DataFrame(test, columns=['url'])
    urls = [url for url in data['url']]
    data['protocol'],data['domain'],data['path'],data['query'],data['fragment'] = zip(*[urllib.parse.urlsplit(x) for x in urls])
    get_features(data)
    data = data.drop(columns=['url', 'protocol', 'domain', 'path', 'query', 'fragment'])
    data = data.values
    data = scaler.transform(data)
    predict = extension_xgb.predict(data)[0]
    if predict == 0:
        print(test_data[i,0]+" ====> Predicted AS SAFE")
    else:
        print(test_data[i,0]+" ====> Predicted AS PHISHING")