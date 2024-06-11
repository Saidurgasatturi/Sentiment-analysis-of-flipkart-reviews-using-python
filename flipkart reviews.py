import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud, STOPWORDS

# Load data
data = pd.read_csv("Website-data_master_flipkart_reviews.csv")
print(data.head())
print(data.isnull().sum())

# Import additional libraries
import nltk
import re
nltk.download('stopwords')
stemmer = nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
import string

# Set of stopwords
stopword = set(stopwords.words('english'))

# Function to clean text data
def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text = " ".join(text)
    return text

# Apply cleaning function to reviews
data["Review"] = data["Review"].apply(clean)

# Calculate rating distribution
ratings = data["Rating"].value_counts()
numbers = ratings.index
quantity = ratings.values

# Create pie chart of rating distribution
import plotly.express as px
figure = px.pie(data, values=quantity, names=numbers, hole=0.5)
figure.show()

# Generate word cloud
text = " ".join(i for i in data.Review)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
plt.figure(figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# Download VADER lexicon for sentiment analysis
nltk.download('vader_lexicon')
sentiments = SentimentIntensityAnalyzer()

# Calculate sentiment scores
data["Positive"] = [sentiments.polarity_scores(i)["pos"] for i in data["Review"]]
data["Negative"] = [sentiments.polarity_scores(i)["neg"] for i in data["Review"]]
data["Neutral"] = [sentiments.polarity_scores(i)["neu"] for i in data["Review"]]

# Keep relevant columns
data = data[["Review", "Positive", "Negative", "Neutral"]]
print(data.head())

# Sum sentiment scores
x = sum(data["Positive"])
y = sum(data["Negative"])
z = sum(data["Neutral"])

# Determine overall sentiment
def sentiment_score(a, b, c):
    if (a > b) and (a > c):
        print("Positive 😊")
    elif (b > a) and (b > c):
        print("Negative 😠")
    else:
        print("Neutral 🙂")

sentiment_score(x, y, z)
