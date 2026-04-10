import pandas as pd
from sklearn.metrics import accuracy_score

def calculate_performance(df):
    """Computes precision and directional bias metrics."""
    # 1. Standard Accuracy
    acc = accuracy_score(df['actual_level'], df['predicted_level'])
    
    # 2. Off-by-One Accuracy (High value for Skill Intelligence)
    df['diff'] = abs(df['actual_level'] - df['predicted_level'])
    off_by_one = (df['diff'] <= 1).mean()
    
    # 3. Directional Bias (Aggressive vs Conservative)
    bias = (df['predicted_level'] - df['actual_level']).mean()
    
    return {
        "accuracy": acc,
        "off_by_one": off_by_one,
        "bias": bias
    }

# Logic to load CSV and print the "Performance Report" table here...
