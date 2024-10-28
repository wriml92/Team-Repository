import numpy as np
import pandas as pd
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F

stop_words = set(stopwords.words('english'))                # 불용어

# 데이터셋 클래스 정의
class ReviewDataset(Dataset):
    def __init__(self, reviews, ratings, text_pipeline, label_pipeline):
        self.reviews = reviews.reset_index(drop=True)  # 인덱스 재설정
        self.ratings = ratings.reset_index(drop=True)  # 인덱스 재설정
        self.text_pipeline = text_pipeline
        self.label_pipeline = label_pipeline

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = self.text_pipeline(self.reviews[idx])
        rating = self.label_pipeline(self.ratings[idx])
        return review.clone().detach(), torch.tensor(rating)  # clone().detach() 사용

# 패딩을 추가하는 함수
def collate_fn(batch):
    reviews, labels = zip(*batch)
    reviews_padded = nn.utils.rnn.pad_sequence(reviews, batch_first=True)  # 패딩 추가
    return reviews_padded, torch.tensor(labels)

# 텍스트 전처리 함수
def preprocess_text(text):
    if isinstance(text, float) or text is None:
        return ""  # 결측치 또는 None을 빈 문자열로 처리
    text = text.lower()  # 소문자로 변환
    text = re.sub(r'[^a-zA-Z ]', '', text) # 영문 제외 제거

    text = ' '.join([word for word in text.split() if word not in stop_words])  # 불용어 제거

    return text if len(text) > 0 else None  # 빈 문자열은 None으로 처리

# 데이터 전처리
def preprocess(dataframe):
    dataframe = dataframe[dataframe['score'] != 3]                                   # 3점인 리뷰 제거
    dataframe['label'] = np.select([dataframe.score > 3], [1], default=0)     # 긍정 부정 라벨 컬럼 추가    
    dataframe["content"] = dataframe["content"].apply(preprocess_text)

    dataframe = dataframe[dataframe['content'].notna() & (dataframe['content'].str.len() > 0)]      # None 및 빈 문자열 제거    
    return dataframe

# 데이터 로드
df = preprocess(pd.read_csv("./data/netflix_reviews.csv"))

X_train = df['content']  # 리뷰 텍스트
y_train = df['label']  # 리뷰 점수

# 데이터 분할
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# 토크나이저 정의
tokenizer = get_tokenizer("basic_english")

# 토큰 목록 생성 함수
def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)

# 단어 사전 생성
vocab = build_vocab_from_iterator(yield_tokens(X_train), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

# 텍스트 파이프라인 정의
text_pipeline = lambda x: torch.tensor(vocab(tokenizer(x)), dtype=torch.long)  # LongTensor로 변환

# 라벨 인코더 정의
label_encoder = LabelEncoder()
label_encoder.fit(y_train)

# 라벨 파이프라인 정의
label_pipeline = lambda x: label_encoder.transform([x])[0]

# 하이퍼파라미터 정의
VOCAB_SIZE = len(vocab)
EMBED_DIM = 64
HIDDEN_DIM = 258
OUTPUT_DIM = 2  # 예측할 점수 개수
EPOCHS = 5
BATCH_SIZE = 64

# 훈련 데이터
train_dataset = ReviewDataset(X_train, y_train, text_pipeline, label_pipeline)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# 테스트 데이터
test_dataset = ReviewDataset(X_valid, y_valid, text_pipeline, label_pipeline)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# LSTM 모델 정의
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden, cell) = self.lstm(embedded.unsqueeze(1))
        return self.fc(hidden[-1])  # 마지막 hidden 상태 반환

# LSTM 모델 초기화
lstm_model = LSTMModel(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM)

# 손실 함수와 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(lstm_model.parameters(), lr=0.1)

# LSTM 모델 학습
def train_lstm_model(model, train_loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for reviews, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(reviews)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader)}')

# LSTM 모델 학습 실행
train_lstm_model(lstm_model, train_dataloader, criterion, optimizer, EPOCHS)

# 예측 함수
def predict_review(lstm_model, review):
    # LSTM 모델을 평가 모드로 설정
    lstm_model.eval()
    with torch.no_grad():
        tensor_review = text_pipeline(review).clone().detach().unsqueeze(0)
        output = lstm_model(tensor_review)
        # 소프트맥스를 사용하여 확률 계산
        probabilities = F.softmax(output, dim=1)
        predicted_probabilities = probabilities.squeeze().numpy()  # NumPy 배열로 변환
        
        # 가장 높은 확률의 클래스를 예측
        prediction = output.argmax(1).item()
        
        # 라벨 역변환
        predicted_label = label_encoder.inverse_transform([prediction])[0]
        return predicted_label, predicted_probabilities[1]

def convert_score_to_rating(score):
    if 0 <= score <= 19:
        return 1  # 0~19 -> 1점
    elif 20 <= score <= 39:
        return 2  # 20~39 -> 2점
    elif 40 <= score <= 59:
        return 3  # 40~59 -> 3점
    elif 60 <= score <= 79:
        return 4  # 60~79 -> 4점
    elif 80 <= score <= 100:
        return 5  # 80~100 -> 5점
    else:
        return 0  # 범위를 벗어난 경우 None 반환
    
# 새로운 리뷰에 대한 예측
predicted_score, predicted_probabilities = predict_review(lstm_model, "this app is great")
print(f'Predicted Score: {convert_score_to_rating(predicted_probabilities*100)}')
predicted_score, predicted_probabilities = predict_review(lstm_model, "Love you netflix")
print(f'Predicted Score: {convert_score_to_rating(predicted_probabilities*100)}')
predicted_score, predicted_probabilities = predict_review(lstm_model, "Some parts were good, but overall, it felt average")
print(f'Predicted Score: {convert_score_to_rating(predicted_probabilities*100)}')
predicted_score, predicted_probabilities = predict_review(lstm_model, "I liked it! Great moments, though a bit lacking in some areas")
print(f'Predicted Score: {convert_score_to_rating(predicted_probabilities*100)}')
predicted_score, predicted_probabilities = predict_review(lstm_model, "poor app")
print(f'Predicted Score: {convert_score_to_rating(predicted_probabilities*100)}')