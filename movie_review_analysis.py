import pandas as pd
from textblob import TextBlob
from gensim.utils import simple_preprocess
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# 텍스트 전처리 함수
def preprocess_text(text):
    if isinstance(text, float) or text is None:
        return ""  # 결측치 또는 None을 빈 문자열로 처리
    
    tokens = simple_preprocess(text, deacc=True)  # simple_preprocess 사용
    return ' '.join(tokens) if tokens else None  # 빈 토큰 리스트를 None으로 처리

# 감성 분석을 위한 함수
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

# 데이터 로드
df = pd.read_csv("./data/netflix_reviews.csv")

# 전처리 및 감성 분석
df['content'] = df['content'].apply(preprocess_text).fillna('')
df['sentiment'] = df['content'].apply(get_sentiment)
df['sentiment_label'] = df['sentiment'].apply(lambda x: 'positive' if x > 0.1 else ('negative' if x < -0.1 else 'neutral'))

# 워드클라우드 생성을 위한 준비
stopwords = set(STOPWORDS)
stopwords.update(['netflix', 'movie', 'show', 'time', 'app', 'series', 'phone'])

# 부정 리뷰 모아 워드클라우드 생성
negative_reviews = ' '.join(df[df['sentiment_label'] == 'negative']['content'])
wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords).generate(negative_reviews)

# 결과 출력
plt.figure(figsize=(12,6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Negative Reviews Word Cloud')
plt.show()
