import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob

# Load the CSV file
csv_path = "news_data.csv"  # Replace with your CSV file path
df = pd.read_csv(csv_path)

# Calculate sentiment polarity for each title
df['sentiment'] = df['title'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Classify sentiment
df['sentiment_label'] = df['sentiment'].apply(lambda x: 'positive' if x > 0 else 'negative' if x < 0 else 'neutral')

# Count sentiment categories
sentiment_counts = df['sentiment_label'].value_counts()

# Plot pie chart
plt.figure(figsize=(7, 7))
sentiment_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['gray', 'green', 'red'])
plt.title('Sentiment Distribution of News Titles')
plt.ylabel('')  # Hide y-label for pie chart
plt.show()

# Print sentiment summary
print("Sentiment Distribution:")
print(sentiment_counts)