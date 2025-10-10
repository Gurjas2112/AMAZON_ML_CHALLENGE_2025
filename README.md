# Amazon ML Challenge 2025 - Smart Product Pricing Solution

🏆 **A comprehensive machine learning solution for predicting e-commerce product prices using catalog content and multimodal feature engineering.**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange.svg)](https://scikit-learn.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-green.svg)](https://lightgbm.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 Challenge Overview

The Smart Product Pricing Challenge focuses on developing an ML solution that analyzes product details and predicts optimal pricing for e-commerce products. The relationship between product attributes and pricing is complex, involving factors like brand recognition, product specifications, and pack quantities.

### Key Metrics
- **Evaluation**: SMAPE (Symmetric Mean Absolute Percentage Error)
- **Best Performance**: **65.81% SMAPE** (LightGBM)
- **Dataset**: 75,000 training + 75,000 test products
- **Features**: 1,012 engineered features from text and structured data

## 📊 Dataset Description

| Column | Type | Description |
|--------|------|-------------|
| `sample_id` | int64 | Unique identifier for each product |
| `catalog_content` | object | Product title, description, and Item Pack Quantity (IPQ) |
| `image_link` | object | URL to product image |
| `price` | float64 | Target variable (training only) |

### Data Statistics
- **Price Range**: $0.13 - $2,796.00
- **Mean Price**: $23.65
- **Median Price**: $14.00
- **Outliers**: 7.37% of products (>$61.39)

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run Complete Solution
```bash
# Train model and generate predictions
python smart_product_pricer.py

# Validate output
python validate_output.py

# Test on sample data
python test_sample.py

# View solution summary
python final_summary.py
```

## 🏗️ Solution Architecture

```
Input: Catalog Content + Image Links
    ↓
📝 Feature Engineering:
├── Structured Text Parsing (Item Name, Value, Unit, Bullets)
├── TF-IDF Vectorization (1000 features, 1-2 grams)
├── Brand Extraction & Categorical Encoding
└── Text Quality Indicators
    ↓
🤖 Model Ensemble:
├── LightGBM (Best: 65.81% SMAPE)
├── XGBoost (66.09% SMAPE)
├── Gradient Boosting (66.23% SMAPE)
├── Random Forest (69.75% SMAPE)
└── Ridge Regression (70.34% SMAPE)
    ↓
📈 Output: Predicted Prices
```

## 🔧 Feature Engineering Pipeline

### 1. Structured Text Features (12 features)
- **Product Attributes**: value, unit, brand, bullet_count
- **Text Metrics**: text_length, word_count, description_length
- **Quality Indicators**: has_description, has_bullets, has_value, has_unit
- **Semantic Features**: price_keyword_count

### 2. NLP Features (1000 features)
- **TF-IDF Vectorization**: 1000-dimensional sparse matrix
- **N-gram Analysis**: Unigrams and bigrams
- **Text Preprocessing**: Cleaning, tokenization, lemmatization

### 3. Data Preprocessing
- **Feature Scaling**: StandardScaler normalization
- **Categorical Encoding**: LabelEncoder with unknown handling
- **Missing Value**: Robust handling with fallback values

## 📈 Model Performance

| Model | SMAPE Score | Description |
|-------|-------------|-------------|
| **LightGBM** | **65.81%** | Best performing gradient boosting |
| XGBoost | 66.09% | Extreme gradient boosting |
| Gradient Boosting | 66.23% | Scikit-learn implementation |
| Random Forest | 69.75% | Ensemble decision trees |
| Ridge Regression | 70.34% | Linear regression with L2 regularization |

### Validation Strategy
- **Train/Validation Split**: 80/20
- **Cross-Validation**: 5-fold for model selection
- **Metric Optimization**: Custom SMAPE implementation
- **Prediction Range**: $0.19 - $230.25 (all positive)

## 📁 Project Structure

```
AMAZON_ML_CHALLENGE_2025/
├── 📄 README.md                              # This file
├── 📋 requirements.txt                       # Python dependencies
├── 📊 test_out.csv                           # Final predictions (75K samples)
├── 📄 Documentation.md                       # Technical documentation
│
├── 🐍 Core Solution
│   ├── smart_product_pricer.py               # Main ML pipeline
│   ├── validate_output.py                    # Output validation
│   ├── test_sample.py                        # Sample testing
│   └── final_summary.py                      # Solution summary
│
├── 📓 Analysis
│   ├── smart_product_pricing_challenge.ipynb # Data exploration
│   └── exploration.py                        # EDA script
│
├── 💾 Models & Data
│   ├── models/smart_product_pricer.pkl       # Trained model artifacts
│   ├── data/                                 # Processed data cache
│   └── images/                               # Downloaded images (future)
│
├── 📈 Results
│   ├── our_sample_predictions.csv            # Sample predictions
│   └── feature_importance.csv                # Feature analysis
│
└── 📂 Dataset
    └── 68e8d1d70b66d_student_resource/
        └── student_resource/
            ├── dataset/
            │   ├── train.csv                  # Training data (75K)
            │   ├── test.csv                   # Test data (75K)
            │   ├── sample_test.csv            # Sample test (100)
            │   └── sample_test_out.csv        # Sample output
            └── src/
                └── utils.py                   # Image download utilities
```

## 🛠️ Technical Implementation

### SmartProductPricer Class
```python
# Core functionality
pricer = SmartProductPricer()
pricer.extract_features(df, is_training=True)
pricer.train_models(X, y)
pricer.predict(X_test)
pricer.save_model("models/smart_product_pricer.pkl")
```

### Key Features
- ✅ **Automated Feature Engineering**: Structured text parsing + TF-IDF
- ✅ **Model Ensemble**: 5 different algorithms with auto-selection
- ✅ **SMAPE Optimization**: Custom metric implementation
- ✅ **Robust Validation**: Format compliance + coverage checks
- ✅ **Production Ready**: Serializable models with preprocessing

## 📊 Results & Validation

### Final Predictions
- **Total Samples**: 75,000 (100% coverage)
- **Price Range**: $0.19 - $230.25
- **Mean Prediction**: $23.57
- **Format Compliance**: ✅ Perfect match with requirements

### Quality Checks
- ✅ All predictions are positive
- ✅ No missing values
- ✅ Correct CSV format (sample_id, price)
- ✅ All test sample_ids covered
- ✅ Reasonable price distribution

## 🔍 Feature Importance Analysis

**Top 5 Most Important Features** (LightGBM):
1. `tfidf_695` - TF-IDF feature (specific product terms)
2. `text_length` - Total character count in catalog
3. `tfidf_949` - TF-IDF feature (product descriptors)
4. `tfidf_605` - TF-IDF feature (brand/category terms)
5. `tfidf_113` - TF-IDF feature (pricing keywords)

*Full analysis available in `feature_importance.csv`*

## 🎯 Academic Integrity

✅ **Compliant Solution**: Uses only provided training data  
✅ **No External Lookup**: No web scraping or external price APIs  
✅ **Open Source Models**: MIT/Apache 2.0 licensed libraries only  
✅ **Reproducible**: All code and models provided  

## 🚀 Usage Examples

### Basic Prediction
```python
from smart_product_pricer import SmartProductPricer
import pandas as pd

# Load and predict
pricer = SmartProductPricer()
pricer.load_model("models/smart_product_pricer.pkl")

test_data = pd.read_csv("test.csv")
predictions = pricer.predict(pricer.extract_features(test_data, False))
```

### Validation
```python
# Validate output format
python validate_output.py

# Expected output:
# ✓ 75,000 predictions generated
# ✓ All prices positive
# ✓ Perfect format compliance
# 🚀 READY FOR SUBMISSION!
```

## 📋 Requirements

### Python Dependencies
- **Core ML**: pandas, numpy, scikit-learn
- **Gradient Boosting**: xgboost, lightgbm
- **NLP**: nltk, TfidfVectorizer
- **Visualization**: matplotlib, seaborn
- **Utilities**: tqdm, pickle

### System Requirements
- **Python**: 3.11+
- **Memory**: 8GB+ recommended
- **Storage**: 500MB for models and data
- **Runtime**: ~5 minutes for full training

## 🏆 Competition Details

- **Challenge**: Amazon ML Challenge 2025
- **Task**: Smart Product Pricing
- **Metric**: SMAPE (lower is better)
- **Dataset**: 150,000 total products
- **Constraint**: No external price lookup allowed
- **Model Limit**: Up to 8B parameters (MIT/Apache 2.0)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

While this is a competition submission, the techniques and approaches can be useful for similar e-commerce pricing challenges:

1. **Feature Engineering**: Structured text parsing methods
2. **Model Ensemble**: Multiple algorithm comparison framework
3. **Validation Pipeline**: Comprehensive output checking
4. **SMAPE Optimization**: Custom metric implementation

## 📞 Contact

For questions about the technical approach or implementation details, please refer to the comprehensive documentation in `Documentation.md`.

---

**🎉 Ready for Submission!** All requirements met and validated. ✅