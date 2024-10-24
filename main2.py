# 도전 과제 - 영화 리뷰 감성 분석

import seaborn as sns  # 그래프를 그리기 위한 seaborn 라이브러리 임포트 (없으면 설치 바랍니다)
import matplotlib.pyplot as plt # 그래프 표시를 위한 pyplot

# 1. 데이터셋 불러오기
import pandas as pd
import re

# 데이터셋 불러오기
file_path = '~/downloads/netflix_reviews.csv'
df = pd.read_csv(file_path)

# 상단 5개, 하단 5개 데이터 확인
head_data = df.head()
tail_data = df.tail()

# 컬럼 정보 및 shape 확인
columns_info = df.columns
shape_info = df.shape

# 2. 데이터 전처리
# 전처리 함수 정의
def preprocess_text(text):
    if isinstance(text, float):
        return ""
    text = text.lower()  # 대문자를 소문자로
    text = re.sub(r'[^\w\s]', '', text)  # 구두점 제거
    text = re.sub(r'\d+', '', text)  # 숫자 제거
    text = text.strip()  # 띄어쓰기 제외하고 빈 칸 제거
    return text

# 'content' 컬럼에 대해 전처리 적용
df['content'] = df['content'].apply(preprocess_text)

# 결과 출력
head_data, tail_data, columns_info, shape_info

# 3. feature 분석 (EDA)

# 리뷰 점수를 기반으로 한 점수별 개수
score_counts = df['score'].value_counts().sort_index()

x=score_counts.index # x축 리뷰점수
y=score_counts.values # y축 리뷰개수

colors = ['#3274A2', '#E0812B', '#3A9239', '#BF3D3E', '#9472B2']  # 그래프바 색상 리스트

plt.figure(figsize=(8, 6))  # 그래프 크기
plt.bar(x, y, color=colors) # x,y값 그래프 색상
plt.xlabel('Score')         # 라벨
plt.ylabel('Count')         # 라벨
plt.title('Distribution of Scores')
plt.show()
# 4. 리뷰 예측 모델 학습시키기 (LSTM)