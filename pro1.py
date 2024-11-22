import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import string

data = {
    'text': [
        "The economy is going to improve, says the finance minister",
        "Click here to win a free iPhone now!",
        "Scientists discover new planet that could sustain life",
        "Breaking: You won't believe what this celebrity did!",
        "Government announces new policy to combat unemployment",
        "Shocking news: This product will change your life!",
        "Researchers develop vaccine for a new strain of virus",
        "Earn $5000 working from home - No experience needed!"
    ],
    'label': ['real', 'fake', 'real', 'fake', 'real', 'fake', 'real', 'fake']
}

df = pd.DataFrame(data)

print(df.info())
print(df.head())
print(df.isnull().sum())

sns.countplot(x='label', data=df)
plt.title("Fake vs Real News Count")
plt.show()

def clean_text(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    return text

df['text'] = df['text'].apply(clean_text)

tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X = tfidf.fit_transform(df['text']).toarray()
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, display_labels=model.classes_)
plt.title("Confusion Matrix")
plt.show()
