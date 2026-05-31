from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# 데이터 불러오기
wine = load_wine()

df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['target'] = wine.target

print("데이터 크기:", df.shape)
print("클래스:", wine.target_names)
print(df['target'].value_counts())
print(df.head())

# train/test 분리
X = wine.data
y = wine.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\ntrain 크기:", X_train.shape)
print("test 크기:", X_test.shape)

# 모델 학습 (n_estimators=50으로 변경해서 비교)
for n in [50, 100, 200]:
    model = RandomForestClassifier(n_estimators=n, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\nn_estimators={n} 정확도: {accuracy_score(y_test, y_pred):.4f}")

# 최종 모델 (n_estimators=100)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\n=== 최종 모델 결과 ===")
print("정확도:", accuracy_score(y_test, y_pred))
print("\n분류 보고서:\n", classification_report(y_test, y_pred, target_names=wine.target_names))
