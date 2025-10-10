# Amazon ML Challenge 2025 - Smart Product Pricing Solution
# Comprehensive ML Pipeline for Product Price Prediction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import re
import pickle
from pathlib import Path

# ML libraries
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import lightgbm as lgb

# NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Other utilities
from tqdm import tqdm
import json
from datetime import datetime

warnings.filterwarnings('ignore')

class SmartProductPricer:
    def __init__(self):
        self.models = {}
        self.vectorizers = {}
        self.scalers = {}
        self.feature_extractors = {}
        self.best_model = None
        self.best_score = float('inf')
        
        # Download NLTK data if needed
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            print("Downloading NLTK data...")
            nltk.download('punkt')
            nltk.download('punkt_tab')
            nltk.download('stopwords')
            nltk.download('wordnet')
            
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def calculate_smape(self, y_true, y_pred):
        """Calculate Symmetric Mean Absolute Percentage Error"""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Ensure predictions are positive
        y_pred = np.maximum(y_pred, 0.01)
        
        # Avoid division by zero
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        denominator = np.where(denominator == 0, 1e-8, denominator)
        
        smape = np.mean(np.abs(y_true - y_pred) / denominator) * 100
        return smape
    
    def extract_text_features(self, text):
        """Extract structured features from catalog content"""
        features = {}
        
        # Extract item name
        name_match = re.search(r'Item Name:\\s*([^\\n]*)', text)
        features['item_name'] = name_match.group(1).strip() if name_match else ""
        
        # Extract value and unit
        value_match = re.search(r'Value:\\s*([0-9.]+)', text)
        features['value'] = float(value_match.group(1)) if value_match else 0.0
        
        unit_match = re.search(r'Unit:\\s*([^\\n]*)', text)
        features['unit'] = unit_match.group(1).strip() if unit_match else ""
        
        # Count bullet points
        features['bullet_count'] = len(re.findall(r'Bullet Point \\d+:', text))
        
        # Extract product description length
        desc_match = re.search(r'Product Description:\\s*(.*)', text, re.DOTALL)
        features['description_length'] = len(desc_match.group(1).strip()) if desc_match else 0
        
        # Extract brand information (first word of item name)
        if features['item_name']:
            features['brand'] = features['item_name'].split()[0] if features['item_name'].split() else ""
        else:
            features['brand'] = ""
        
        # Text quality indicators
        features['has_description'] = 1 if 'Product Description:' in text else 0
        features['has_bullets'] = 1 if 'Bullet Point' in text else 0
        features['has_value'] = 1 if value_match else 0
        features['has_unit'] = 1 if unit_match else 0
        
        # Basic text statistics
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        
        # Price-related keywords
        price_keywords = ['premium', 'luxury', 'professional', 'deluxe', 'organic', 'natural']
        features['price_keyword_count'] = sum(1 for keyword in price_keywords if keyword.lower() in text.lower())
        
        return features
    
    def preprocess_text(self, text):
        """Clean and preprocess text for NLP features"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep important ones
        text = re.sub(r'[^a-zA-Z0-9\\s\\.]', ' ', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def extract_features(self, df, is_training=True):
        """Extract all features from the dataset"""
        print("Extracting features...")
        
        # Extract text features
        text_features = df['catalog_content'].apply(self.extract_text_features)
        text_features_df = pd.DataFrame(text_features.tolist())
        
        # Combine with original data
        features_df = pd.concat([df, text_features_df], axis=1)
        
        # Preprocess text for TF-IDF
        features_df['cleaned_text'] = df['catalog_content'].apply(self.preprocess_text)
        
        if is_training:
            # Fit TF-IDF vectorizer
            self.vectorizers['tfidf'] = TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )
            tfidf_features = self.vectorizers['tfidf'].fit_transform(features_df['cleaned_text'])
            
            # Create TF-IDF feature names
            tfidf_feature_names = [f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
            
            # Encode categorical features
            categorical_features = ['unit', 'brand']
            for feature in categorical_features:
                if feature in features_df.columns:
                    self.feature_extractors[f'{feature}_encoder'] = LabelEncoder()
                    features_df[f'{feature}_encoded'] = self.feature_extractors[f'{feature}_encoder'].fit_transform(features_df[feature].fillna('unknown'))
        
        else:
            # Transform using fitted vectorizer
            tfidf_features = self.vectorizers['tfidf'].transform(features_df['cleaned_text'])
            tfidf_feature_names = [f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
            
            # Encode categorical features using fitted encoders
            categorical_features = ['unit', 'brand']
            for feature in categorical_features:
                if feature in features_df.columns and f'{feature}_encoder' in self.feature_extractors:
                    encoder = self.feature_extractors[f'{feature}_encoder']
                    # Handle unseen categories
                    unique_values = set(encoder.classes_)
                    features_df[f'{feature}_encoded'] = features_df[feature].fillna('unknown').apply(
                        lambda x: encoder.transform([x])[0] if x in unique_values else len(encoder.classes_) - 1
                    )
        
        # Convert TF-IDF to DataFrame
        tfidf_df = pd.DataFrame(tfidf_features.toarray(), columns=tfidf_feature_names, index=features_df.index)
        
        # Select numerical features
        numerical_features = [
            'value', 'bullet_count', 'description_length', 'has_description',
            'has_bullets', 'has_value', 'has_unit', 'text_length', 'word_count',
            'price_keyword_count', 'unit_encoded', 'brand_encoded'
        ]
        
        # Filter existing features
        available_features = [f for f in numerical_features if f in features_df.columns]
        numerical_df = features_df[available_features]
        
        # Combine all features
        final_features = pd.concat([numerical_df, tfidf_df], axis=1)
        
        # Scale features if training
        if is_training:
            self.scalers['features'] = StandardScaler()
            final_features_scaled = self.scalers['features'].fit_transform(final_features)
        else:
            final_features_scaled = self.scalers['features'].transform(final_features)
        
        return pd.DataFrame(final_features_scaled, columns=final_features.columns, index=features_df.index)
    
    def train_models(self, X, y):
        """Train multiple models and select the best one"""
        print("Training models...")
        
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Define models
        models_config = {
            'xgboost': xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            ),
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ),
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'ridge': Ridge(alpha=1.0)
        }
        
        # Train and evaluate models
        for name, model in models_config.items():
            print(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_val)
            
            # Ensure positive predictions
            y_pred = np.maximum(y_pred, 0.01)
            
            # Calculate SMAPE
            smape = self.calculate_smape(y_val, y_pred)
            
            print(f"{name} - SMAPE: {smape:.4f}")
            
            # Store model
            self.models[name] = model
            
            # Update best model
            if smape < self.best_score:
                self.best_score = smape
                self.best_model = name
        
        print(f"\\nBest model: {self.best_model} with SMAPE: {self.best_score:.4f}")
    
    def predict(self, X):
        """Make predictions using the best model"""
        if self.best_model is None:
            raise ValueError("No model has been trained yet!")
        
        predictions = self.models[self.best_model].predict(X)
        
        # Ensure positive predictions
        predictions = np.maximum(predictions, 0.01)
        
        return predictions
    
    def save_model(self, filepath):
        """Save the trained model and preprocessors"""
        model_data = {
            'models': self.models,
            'vectorizers': self.vectorizers,
            'scalers': self.scalers,
            'feature_extractors': self.feature_extractors,
            'best_model': self.best_model,
            'best_score': self.best_score
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model and preprocessors"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.models = model_data['models']
        self.vectorizers = model_data['vectorizers']
        self.scalers = model_data['scalers']
        self.feature_extractors = model_data['feature_extractors']
        self.best_model = model_data['best_model']
        self.best_score = model_data['best_score']
        
        print(f"Model loaded from {filepath}")
        print(f"Best model: {self.best_model} with SMAPE: {self.best_score:.4f}")


def main():
    """Main training pipeline"""
    # Data paths
    DATASET_PATH = "68e8d1d70b66d_student_resource/student_resource/dataset/"
    
    # Initialize the model
    pricer = SmartProductPricer()
    
    # Load data
    print("Loading training data...")
    train_df = pd.read_csv(os.path.join(DATASET_PATH, "train.csv"))
    print(f"Training data shape: {train_df.shape}")
    
    # Extract features
    X = pricer.extract_features(train_df, is_training=True)
    y = train_df['price']
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    
    # Train models
    pricer.train_models(X, y)
    
    # Save model
    os.makedirs("models", exist_ok=True)
    pricer.save_model("models/smart_product_pricer.pkl")
    
    # Load test data and make predictions
    print("\\nLoading test data...")
    test_df = pd.read_csv(os.path.join(DATASET_PATH, "test.csv"))
    print(f"Test data shape: {test_df.shape}")
    
    # Extract test features
    X_test = pricer.extract_features(test_df, is_training=False)
    
    # Make predictions
    print("Making predictions...")
    predictions = pricer.predict(X_test)
    
    # Create output dataframe
    output_df = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': predictions
    })
    
    # Save predictions
    output_df.to_csv("test_out.csv", index=False)
    print(f"Predictions saved to test_out.csv")
    print(f"Total predictions: {len(output_df)}")
    print(f"Prediction range: ${predictions.min():.2f} - ${predictions.max():.2f}")
    print(f"Mean prediction: ${predictions.mean():.2f}")


if __name__ == "__main__":
    main()