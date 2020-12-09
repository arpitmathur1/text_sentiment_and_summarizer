import nltk
from nltk.corpus import stopwords
import pandas as pd

import random
random.seed(0)


train = pd.read_csv("news_summary_more.csv")
print("before removing duplicates", len(train))

# drop some duplicates
train = train.drop_duplicates(subset='text', keep="last")

print("after removing duplicates", len(train))
train = train.reset_index(drop=True)
print(train.head())