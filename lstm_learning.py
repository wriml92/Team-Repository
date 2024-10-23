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
    text = re.sub(r'[^\w\s]', '', text)  # 구두점 제거
    text = re.sub(r'\d+', '', text)  # 숫자 제거
    text = text.strip()  # 앞뒤 공백 제거
    return text if len(text) > 0 else None  # 빈 문자열은 None으로 처리

# 데이터 로드
df = pd.read_csv("./netflix_reviews.csv")

df["content"] = df["content"].apply(preprocess_text)
# None 및 빈 문자열 제거
df = df[df['content'].notna() & (df['content'].str.len() > 0)]
reviews = df['content']  # 리뷰 텍스트
ratings = df['score']  # 리뷰 점수

# 데이터 분할
train_reviews, test_reviews, train_ratings, test_ratings = train_test_split(reviews, ratings, test_size=0.2, random_state=42)

# 토크나이저 정의
tokenizer = get_tokenizer("basic_english")

# 토큰 목록 생성 함수
def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)

# 단어 사전 생성
vocab = build_vocab_from_iterator(yield_tokens(train_reviews), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

# 텍스트 파이프라인 정의
text_pipeline = lambda x: torch.tensor(vocab(tokenizer(x)), dtype=torch.long)  # LongTensor로 변환

# 라벨 인코더 정의
label_encoder = LabelEncoder()
label_encoder.fit(train_ratings)

# 라벨 파이프라인 정의
label_pipeline = lambda x: label_encoder.transform([x])[0]

# 데이터셋 정의
train_dataset = ReviewDataset(train_reviews, train_ratings, text_pipeline, label_pipeline)
test_dataset = ReviewDataset(test_reviews, test_ratings, text_pipeline, label_pipeline)

# 데이터 로더 정의
BATCH_SIZE = 64
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# LSTM 모델 정의
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden, cell) = self.lstm(embedded.unsqueeze(1))  # (batch, seq, feature)
        return self.fc(hidden[-1])

# 하이퍼파라미터 정의
VOCAB_SIZE = len(vocab)
EMBED_DIM = 64
HIDDEN_DIM = 128
OUTPUT_DIM = len(label_encoder.classes_)  # 점수 개수

# LSTM 모델 초기화
lstm_model = LSTMModel(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM)

# 손실 함수와 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(lstm_model.parameters(), lr=0.01)

# LSTM 모델 학습
def train_lstm_model(model, train_loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        for reviews, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(reviews)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

# LSTM 모델 학습 실행
train_lstm_model(lstm_model, train_dataloader, criterion, optimizer)

# LSTM 모델로부터 특징 추출
def extract_features(model, data_loader):
    model.eval()
    features = []
    with torch.no_grad():
        for reviews, _ in data_loader:
            outputs = model(reviews)
            features.append(outputs)
    return torch.cat(features)

# 학습 데이터에서 특징 추출
X_train_features = extract_features(lstm_model, train_dataloader).numpy()
y_train = train_ratings.tolist()  # 원래 점수로 변환

# 로지스틱 회귀 모델 생성 및 학습
logistic_model = LogisticRegression()
logistic_model.fit(X_train_features, y_train)

# 테스트 데이터에서 특징 추출
X_test_features = extract_features(lstm_model, test_dataloader).numpy()
y_test = test_ratings.tolist()

# 예측
# y_pred = logistic_model.predict(X_test_features)

# # 평가
# print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
# print(f"Classification Report:\n{classification_report(y_test, y_pred, zero_division=1)}")
# print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

# 예측 함수
def predict_review(logistic_model, model, review):
    model.eval()
    with torch.no_grad():
        tensor_review = text_pipeline(review).clone().detach().unsqueeze(0)
        features = model(tensor_review).numpy()
        prediction = logistic_model.predict(features)
        return label_encoder.inverse_transform(prediction)[0]

# 새로운 리뷰에 대한 예측
new_review = "this app it's so very nice"
predicted_score = predict_review(logistic_model, lstm_model, new_review)
print(f'Predicted Score: {predicted_score}')
