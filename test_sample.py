# Test on Sample Data - Amazon ML Challenge 2025

import pandas as pd
import numpy as np
import pickle

def test_sample_predictions():
    """Test our model on sample data and compare performance"""
    
    # Load sample test data
    sample_test_df = pd.read_csv("68e8d1d70b66d_student_resource/student_resource/dataset/sample_test.csv")
    sample_out_df = pd.read_csv("68e8d1d70b66d_student_resource/student_resource/dataset/sample_test_out.csv")
    
    print("Testing on Sample Data")
    print("="*40)
    print(f"Sample test data shape: {sample_test_df.shape}")
    print(f"Sample output shape: {sample_out_df.shape}")
    
    # Load our trained model
    try:
        from smart_product_pricer import SmartProductPricer
        pricer = SmartProductPricer()
        pricer.load_model("models/smart_product_pricer.pkl")
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Extract features for sample data
    try:
        X_sample = pricer.extract_features(sample_test_df, is_training=False)
        print(f"✓ Features extracted: {X_sample.shape}")
    except Exception as e:
        print(f"✗ Error extracting features: {e}")
        return
    
    # Make predictions
    try:
        predictions = pricer.predict(X_sample)
        print(f"✓ Predictions generated: {len(predictions)}")
    except Exception as e:
        print(f"✗ Error making predictions: {e}")
        return
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'sample_id': sample_test_df['sample_id'],
        'our_prediction': predictions,
        'sample_prediction': sample_out_df['price']
    })
    
    # Compare our predictions with sample predictions
    print("\\nPrediction Comparison:")
    print("="*40)
    print(results_df.head(10))
    
    # Calculate basic statistics
    print("\\nStatistics Comparison:")
    print("="*40)
    print(f"Our predictions - Min: ${predictions.min():.2f}, Max: ${predictions.max():.2f}, Mean: ${predictions.mean():.2f}")
    print(f"Sample predictions - Min: ${sample_out_df['price'].min():.2f}, Max: ${sample_out_df['price'].max():.2f}, Mean: ${sample_out_df['price'].mean():.2f}")
    
    # Save our sample predictions
    our_sample_output = pd.DataFrame({
        'sample_id': sample_test_df['sample_id'],
        'price': predictions
    })
    
    our_sample_output.to_csv("our_sample_predictions.csv", index=False)
    print(f"\\n✓ Our sample predictions saved to our_sample_predictions.csv")
    
    # Basic validation
    print("\\nValidation Checks:")
    print("="*40)
    
    # Check all positive
    if all(predictions > 0):
        print("✓ All predictions are positive")
    else:
        print("✗ Some predictions are not positive")
    
    # Check reasonable range
    if 0.1 <= predictions.min() and predictions.max() <= 1000:
        print("✓ Predictions are in reasonable range")
    else:
        print("⚠ Predictions may be outside expected range")
    
    # Check format
    if len(predictions) == len(sample_test_df):
        print("✓ Correct number of predictions")
    else:
        print("✗ Incorrect number of predictions")
    
    return results_df

def analyze_feature_importance():
    """Analyze which features are most important for predictions"""
    
    try:
        # Load the trained model
        with open("models/smart_product_pricer.pkl", 'rb') as f:
            model_data = pickle.load(f)
        
        best_model_name = model_data['best_model']
        best_model = model_data['models'][best_model_name]
        
        print(f"\\nFeature Importance Analysis ({best_model_name}):")
        print("="*50)
        
        # For tree-based models, get feature importance
        if hasattr(best_model, 'feature_importances_'):
            # Create dummy features to get feature names
            dummy_data = pd.DataFrame({'catalog_content': ['dummy text'], 'image_link': ['dummy_link']})
            from smart_product_pricer import SmartProductPricer
            pricer = SmartProductPricer()
            pricer.vectorizers = model_data['vectorizers']
            pricer.scalers = model_data['scalers']
            pricer.feature_extractors = model_data['feature_extractors']
            
            dummy_features = pricer.extract_features(dummy_data, is_training=False)
            feature_names = dummy_features.columns
            
            # Get feature importance
            importance = best_model.feature_importances_
            
            # Create importance dataframe
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            print("Top 20 Most Important Features:")
            print(importance_df.head(20).to_string(index=False))
            
            # Save feature importance
            importance_df.to_csv("feature_importance.csv", index=False)
            print(f"\\n✓ Feature importance saved to feature_importance.csv")
            
        else:
            print("Feature importance not available for this model type")
            
    except Exception as e:
        print(f"Error analyzing feature importance: {e}")

if __name__ == "__main__":
    # Test on sample data
    results = test_sample_predictions()
    
    # Analyze feature importance
    analyze_feature_importance()
    
    print("\\n" + "="*50)
    print("🎯 SAMPLE TESTING COMPLETED! 🎯")
    print("="*50)