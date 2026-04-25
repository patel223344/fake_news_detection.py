# Fake News Detection

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

data = {
    'news':[
        'Breaking: major event happened',
        'You won a lottery click here',
        'Government announces new policy',
        'Earn money fast online'
    ],
    'label':[0,1,0,1]  # 0=real,1=fake
}

df = pd.DataFrame(data)

vec = TfidfVectorizer()
X = vec.fit_transform(df['news'])
y = df['label']

model = LogisticRegression()
model.fit(X,y)

print(model.predict(vec.transform(['earn money now'])))
