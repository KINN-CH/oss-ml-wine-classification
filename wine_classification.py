from sklearn.datasets import load_wine
import pandas as pd

# 데이터 불러오기
wine = load_wine()

df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['target'] = wine.target

print("데이터 크기:", df.shape)
print("클래스:", wine.target_names)
print(df['target'].value_counts())
print(df.head())
