import pandas as pd
import numpy as np
from textblob import TextBlob
from wordcloud import WordCloud
import re
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the data
data = pd.read_csv("C:\\Users\\yamini\\OneDrive\\Desktop\\flipkart.csv")
print(data.head(10))
print(data.isnull().sum())

def cleanReviews(text):
    text = re.sub('@[A-Za-z0-9_]+', '', text)
    text = re.sub('#','',text) 
    text = re.sub('https?:\/\/\S+', '', text)  
    text = re.sub('\n',' ',text) 
    text = re.sub(r'www\S+', " ", text) 
    text = re.sub(r'\.|/|:|-', " ", text)
    text = re.sub(r'[^\w\s]','',text)
    return text
data['cleanedReviews'] = data['Review'].apply(cleanReviews) 
data.head()

data_cleaned=data[['cleanedReviews','Rating']]
data_cleaned

rating_counts = data_cleaned['Rating'].value_counts().sort_index()
print(rating_counts)
x = rating_counts.index

plt.bar(x, rating_counts)

plt.xlabel('Rating')
plt.ylabel('Count')
plt.title('Rating Distribution')

plt.show() 

def getAnalysis(rating):
    if rating < 3:
        return 'Negative'
    elif rating == 3:
        return 'Neutral'
    else:
        return 'Positive'
data_cleaned['Analysis'] = data_cleaned['Rating'].apply(getAnalysis)
print(data_cleaned)

X = df['cleanedReviews']
y = df['Analysis']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Initialize models
svm_model = SVC(kernel='linear', probability=True)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

# Train models
svm_model.fit(X_train_tfidf, y_train)
rf_model.fit(X_train_tfidf, y_train)
gb_model.fit(X_train_tfidf, y_train)

# Predict probabilities for each class
proba_svm = svm_model.predict_proba(X_test_tfidf)
proba_rf = rf_model.predict_proba(X_test_tfidf)
proba_gb = gb_model.predict_proba(X_test_tfidf)

# Get predictions from models
y_pred_svm = svm_model.predict(X_test_tfidf)
y_pred_rf = rf_model.predict(X_test_tfidf)
y_pred_gb = gb_model.predict(X_test_tfidf)

# Convert the predictions to a NumPy array if they are not already
predictions = np.array([y_pred_svm, y_pred_rf, y_pred_gb]).T

# Defining class labels explicitly
class_labels = ['Positive', 'Neutral', 'Negative']

# Apply majority voting
final_predictions = []
for pred in predictions:
    pred_counts = Counter(pred)
    # Use majority voting
    most_common_label, count = pred_counts.most_common(1)[0]
    final_predictions.append(most_common_label)

final_predictions = np.array(final_predictions)

# Print confusion matrix and classification report
print("True labels distribution:", Counter(y_test))
print("Predicted labels distribution:", Counter(final_predictions))

print("Confusion Matrix:\n", confusion_matrix(y_test, final_predictions, labels=class_labels))
print("Classification Report:\n", classification_report(y_test, final_predictions, labels=class_labels, zero_division=0))

# Calculate the percentage of each sentiment
positive_reviews = np.sum(final_predictions == 'Positive')
negative_reviews = np.sum(final_predictions == 'Negative')
neutral_reviews = np.sum(final_predictions == 'Neutral')
total_reviews = len(final_predictions)

print(f"Percentage of Positive reviews: {positive_reviews / total_reviews * 100:.2f}%")
print(f"Percentage of Negative reviews: {negative_reviews / total_reviews * 100:.2f}%")
print(f"Percentage of Neutral reviews: {neutral_reviews / total_reviews * 100:.2f}%")

# Accuracy score
accuracy = accuracy_score(y_test, final_predictions)
print("Final Majority Voting Accuracy:", accuracy)




