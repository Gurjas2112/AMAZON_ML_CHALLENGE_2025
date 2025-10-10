# Amazon ML Challenge 2025 - Data Exploration and Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
from collections import Counter
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pickle
import os

warnings.filterwarnings('ignore')

# Set up style
plt.style.use('default')
sns.set_palette("husl")

# Data paths
DATASET_PATH = "68e8d1d70b66d_student_resource/student_resource/dataset/"
DATA_PATH = "data/"
MODELS_PATH = "models/"

# Create necessary directories
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(MODELS_PATH, exist_ok=True)
os.makedirs("images", exist_ok=True)

# Load data
print("Loading training data...")
train_df = pd.read_csv(os.path.join(DATASET_PATH, "train.csv"))
print(f"Training data shape: {train_df.shape}")

print("\nLoading test data...")
test_df = pd.read_csv(os.path.join(DATASET_PATH, "test.csv"))
print(f"Test data shape: {test_df.shape}")

print("\nLoading sample data...")
sample_test_df = pd.read_csv(os.path.join(DATASET_PATH, "sample_test.csv"))
sample_out_df = pd.read_csv(os.path.join(DATASET_PATH, "sample_test_out.csv"))
print(f"Sample test shape: {sample_test_df.shape}")
print(f"Sample output shape: {sample_out_df.shape}")