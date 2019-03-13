import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('train.csv')
df.dropna(inplace=True)
df.reset_index(drop=True)

li = ['Transaction-Type', 'Complaint-reason', 'Company-response', 'Consumer-disputes']

# contains dummies of features
for item in li:
	dummies_df = pd.concat([df, pd.get_dummies(df[item])], axis=1)

print('dummies are made')

# contains all 4 y cols
y = pd.get_dummies(df['Complaint-Status'])
print('y created')

count_vectorizer = CountVectorizer(stop_words='english')
count_train = count_vectorizer.fit_transform(df['Consumer-complaint-summary'])
print('count_train made')

del df
print('df deleted')

count_df = pd.DataFrame(count_train.A, columns=count_vectorizer.get_feature_names())
print('count_df made')

del count_train
print('count_train deleted')

X_train, X_test, y_train, y_test = train_test_split(pd.concat([dummies_df, count_df], axis=1), y, test_size=0.33, random_state=42)

del dummies_df, count_df
del y
print('X, y deleted')

# training the model
print('training the model')
clf = LinearSVC()
svc=clf.fit(X_train, y_train['Closed'])
pred=svc.predict(X_test)
# metrics  
print('print the performances metrics')
print(confusion_matrix(y_test['Closed'],pred))  
print(classification_report(y_test['Closed'],pred))

