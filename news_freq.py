# Requirements: pandas, matplotlib, textblob
# Install with: pip install pandas matplotlib textblob

import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob

# --- CONFIGURE YOUR CSV PATH AND GROUPING PERIOD ---
csv_path = "news_data.csv"  # Replace with your CSV file path
group_period = 'W'  # 'W' for week, 'D' for day, 'M' for month, etc.

# --- LOAD CSV AND CHECK FOR REQUIRED COLUMNS ---
df = pd.read_csv(csv_path)

# Ensure 'published' and 'title' columns exist
if 'published' not in df.columns or 'title' not in df.columns:
    print("Error: CSV must contain 'published' and 'title' columns.")
    exit()

# --- NEWS FREQUENCY OVER TIME ---
# Convert 'published' to datetime, handling errors
df['published'] = pd.to_datetime(df['published'], errors='coerce')
# Note: If dates are in a specific format, specify it, e.g.,
# df['published'] = pd.to_datetime(df['published'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

# Drop rows with invalid dates
df = df.dropna(subset=['published'])

# Check if there are any articles left after cleaning
if df.empty:
    print("No valid articles found after date cleaning.")
    exit()

# Group by the specified period (e.g., week, day)
news_by_period = df.groupby(df['published'].dt.to_period(group_period)).size()

# Check if there is more than one unique period to plot
unique_periods = news_by_period.index.nunique()
if unique_periods > 1:
    plt.figure(figsize=(10, 5))
    news_by_period.plot(kind='line')
    plt.title('News Articles Over Time')
    plt.xlabel('Period')
    plt.ylabel('Number of Articles')
    plt.show()
else:
    print(f"All articles are from the same period ({group_period}). Cannot plot a time series.")

# Print debug information
print(f"Total articles after date cleaning: {len(df)}")
print(f"Date range: from {df['published'].min()} to {df['published'].max()}")
print(f"Grouped by: {group_period}")
print(news_by_period)

# --- SENTIMENT ANALYSIS ---
# Calculate sentiment polarity for each title
df['sentiment'] = df['title'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Classify sentiment
df['sentiment_label'] = df['sentiment'].apply(lambda x: 'positive' if x > 0 else 'negative' if x < 0 else 'neutral')

# Count sentiment categories
sentiment_counts = df['sentiment_label'].value_counts()

# Plot pie chart
plt.figure(figsize=(7, 7))
sentiment_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['green', 'red', 'gray'])
plt.title('Sentiment Distribution of News Titles')
plt.ylabel('')  # Hide y-label for pie chart
plt.show()

# Print sentiment summary
print("Sentiment Distribution:")
print(sentiment_counts)

# --- OPTIONAL: Test matplotlib (uncomment to test) ---
# plt.plot([1, 2, 3], [4, 5, 6])
# plt.title('Test Plot')
# plt.show()