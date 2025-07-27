import feedparser
import pandas as pd
import os

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.patches as mpatches

from wordcloud import WordCloud
import matplotlib.pyplot as plt

# --- CONFIG ---
csv_path = "news_data.csv"
rss_url = "https://news.google.com/rss/search?q=WAR&hl=en&ceid=GLOBAL:en"
#rss_url = "https://news.google.com/rss?hl=en&ceid=GLOBAL:en"
#rss_url = "https://news.google.com/rss/search?q=world+economy&hl=en&ceid=GLOBAL:en"


# --- FETCH NEW ENTRIES FROM RSS ---
feed = feedparser.parse(rss_url)
new_entries = []

for entry in feed.entries:
    new_entries.append({
        "title": entry.title.strip(),
        "link": entry.link.strip(),
        "published": entry.published if "published" in entry else None
    })

new_df = pd.DataFrame(new_entries)

# --- LOAD EXISTING CSV (IF EXISTS) ---
if os.path.exists(csv_path):
    old_df = pd.read_csv(csv_path)

    # Combine and drop duplicates based on the 'title'
    combined_df = pd.concat([old_df, new_df], ignore_index=True)
    combined_df.drop_duplicates(subset="title", keep="first", inplace=True)
else:
    combined_df = new_df

# --- SAVE BACK TO CSV ---
combined_df.to_csv(csv_path, index=False)
print(f"✅ CSV updated: {len(new_df)} new rows scanned. Total entries: {len(combined_df)}")