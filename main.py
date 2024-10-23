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
# count : 
# mean : 
# std :
# min :
# 25% :
# 50% : 
# max : 

# 각 열의 결측치 갯수 확인
print(titanic.isnull().sum())