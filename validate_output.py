# Validation Script for Amazon ML Challenge 2025 Solution

import pandas as pd
import numpy as np

def validate_output(filepath="test_out.csv", sample_filepath="68e8d1d70b66d_student_resource/student_resource/dataset/sample_test_out.csv"):
    """Validate the output file format and content"""
    
    # Load the generated output
    try:
        output_df = pd.read_csv(filepath)
        print(f"✓ Successfully loaded output file: {filepath}")
        print(f"  Shape: {output_df.shape}")
    except Exception as e:
        print(f"✗ Error loading output file: {e}")
        return False
    
    # Load the sample output for format comparison
    try:
        sample_df = pd.read_csv(sample_filepath)
        print(f"✓ Successfully loaded sample file: {sample_filepath}")
        print(f"  Shape: {sample_df.shape}")
    except Exception as e:
        print(f"✗ Error loading sample file: {e}")
        return False
    
    # Check columns
    expected_columns = ['sample_id', 'price']
    if list(output_df.columns) == expected_columns:
        print("✓ Columns are correct:", output_df.columns.tolist())
    else:
        print(f"✗ Incorrect columns. Expected: {expected_columns}, Got: {output_df.columns.tolist()}")
        return False
    
    # Check data types
    print(f"✓ Data types:")
    print(f"  sample_id: {output_df['sample_id'].dtype}")
    print(f"  price: {output_df['price'].dtype}")
    
    # Check for missing values
    missing_values = output_df.isnull().sum()
    if missing_values.sum() == 0:
        print("✓ No missing values")
    else:
        print(f"✗ Found missing values: {missing_values}")
        return False
    
    # Check price values
    prices = output_df['price']
    
    # Check for positive prices
    negative_prices = (prices <= 0).sum()
    if negative_prices == 0:
        print("✓ All prices are positive")
    else:
        print(f"✗ Found {negative_prices} non-positive prices")
        return False
    
    # Price statistics
    print(f"✓ Price statistics:")
    print(f"  Min: ${prices.min():.2f}")
    print(f"  Max: ${prices.max():.2f}")
    print(f"  Mean: ${prices.mean():.2f}")
    print(f"  Median: ${prices.median():.2f}")
    print(f"  Std: ${prices.std():.2f}")
    
    # Check for reasonable price range (basic sanity check)
    if prices.min() >= 0.01 and prices.max() <= 10000:
        print("✓ Price range appears reasonable")
    else:
        print(f"⚠ Price range may be unusual: ${prices.min():.2f} - ${prices.max():.2f}")
    
    # Check sample_id uniqueness
    if output_df['sample_id'].nunique() == len(output_df):
        print("✓ All sample_ids are unique")
    else:
        print(f"✗ Duplicate sample_ids found")
        return False
    
    # Compare format with sample
    print(f"✓ Format comparison with sample:")
    print(f"  Sample format: {sample_df.dtypes.to_dict()}")
    print(f"  Output format: {output_df.dtypes.to_dict()}")
    
    # Show first few predictions
    print(f"✓ First 10 predictions:")
    print(output_df.head(10).to_string(index=False))
    
    print("\n" + "="*50)
    print("🎉 OUTPUT VALIDATION PASSED! 🎉")
    print("="*50)
    
    return True

def compare_with_test_data():
    """Compare output with test data to ensure all sample_ids are covered"""
    
    # Load test data
    test_df = pd.read_csv("68e8d1d70b66d_student_resource/student_resource/dataset/test.csv")
    output_df = pd.read_csv("test_out.csv")
    
    print("Comparing with test data...")
    
    # Check if all test sample_ids are in output
    test_ids = set(test_df['sample_id'])
    output_ids = set(output_df['sample_id'])
    
    missing_ids = test_ids - output_ids
    extra_ids = output_ids - test_ids
    
    if len(missing_ids) == 0:
        print("✓ All test sample_ids are present in output")
    else:
        print(f"✗ Missing {len(missing_ids)} sample_ids from test data")
        print(f"  First few missing: {list(missing_ids)[:5]}")
    
    if len(extra_ids) == 0:
        print("✓ No extra sample_ids in output")
    else:
        print(f"⚠ Found {len(extra_ids)} extra sample_ids not in test data")
        print(f"  First few extra: {list(extra_ids)[:5]}")
    
    if len(missing_ids) == 0 and len(extra_ids) == 0:
        print("🎯 Perfect match with test data sample_ids!")
    
    return len(missing_ids) == 0

if __name__ == "__main__":
    print("Validating Amazon ML Challenge 2025 Output")
    print("="*50)
    
    # Validate output format
    is_valid = validate_output()
    
    if is_valid:
        print("\n" + "="*50)
        print("Checking coverage of test data...")
        print("="*50)
        
        # Check coverage
        coverage_ok = compare_with_test_data()
        
        if coverage_ok:
            print("\n🚀 READY FOR SUBMISSION! 🚀")
        else:
            print("\n⚠ Please check sample_id coverage before submission")
    else:
        print("\n❌ Please fix validation errors before submission")