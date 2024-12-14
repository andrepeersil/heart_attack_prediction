#%%
import pandas as pd

df = pd.read_csv('./data/heart.csv',)
df
#%%

df_features = df.drop(columns=['output'])
df_features.head()

#%%

X = df[pd.Series(df_features.columns)]
y = df['output']
#%%
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier(random_state=42)

clf.fit(X_train, y_train)

#%%
y_pred = clf.predict(X_test)
#%%

from sklearn.metrics import accuracy_score

print("Acurácia:", accuracy_score(y_test, y_pred))