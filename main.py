# 필수 과제 - 타이타닉 생존자 예측

# 1. 데이터셋 불러오기

# seaborn 모듈 불러오기
import seaborn as sns
titanic = sns.load_dataset('titanic')

# 2. 특징(feature) 분석

# 타이타닉 데이터 셋 첫 5행 출력
print(titanic.head())

# 기본적인 통계 확인
print(titanic.describe())

# describe 함수
# count : 카운트 수(행 갯수)
# mean : 열의 평균값
# std : 열의 표준편차
# min : 열의 최솟값
# 25% : 열의 하위 25%값
# 50% : 열의 중간값
# max : 열의 최댓값

# 각 열의 결측치 갯수 확인
print(titanic.isnull().sum())

# 결측치 처리(age의 결측치를 중간값으로, embarked의 결측치를 최솟값으로)
titanic['age'].fillna(titanic['age'].median(), inplace=True)
titanic['embarked'].fillna(titanic['embarked'].mode()[0], inplace=True)

print(titanic['age'].isnull().sum())
print(titanic['embarked'].isnull().sum())
