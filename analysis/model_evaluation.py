import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import xgboost as xgb

def load_and_clean_data(csv_path):
    df = pd.read_csv(csv_path)

    if "LiquidClassName" not in df.columns:
        df.rename(columns={df.columns[0]: "LiquidClassName"}, inplace=True)

    liquid_classes = df["LiquidClassName"]
    features = df.drop(columns=["LiquidClassName"])

    numeric_features = features.select_dtypes(include=["number"])
    non_numeric_features = features.drop(columns=numeric_features.columns, errors="ignore")
    numeric_features = numeric_features.fillna(numeric_features.median())

    clean_df = pd.concat([liquid_classes, numeric_features, non_numeric_features], axis=1)
    
    # SANITIZE: Remove duplicate rows first thing
    original_len = len(clean_df)
    clean_df = clean_df.drop_duplicates()
    duplicates_removed = original_len - len(clean_df)
    
    print(f"\n=== Data Sanitization ===")
    print(f"Original rows: {original_len}")
    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Unique rows remaining: {len(clean_df)}")
    
    # Update liquid_classes and numeric_features after deduplication
    liquid_classes = clean_df["LiquidClassName"]
    numeric_features = clean_df.drop(columns=["LiquidClassName"]).select_dtypes(include=["number"])
    
    return clean_df, numeric_features, liquid_classes

def split_train_test(df, numeric_features, liquid_classes, test_size=0.2):
    """Split data into train and test sets at the very beginning"""
    train_idx, test_idx = train_test_split(
        df.index, 
        test_size=test_size, 
        random_state=42, 
        shuffle=True
    )
    
    df_train = df.loc[train_idx]
    df_test = df.loc[test_idx]
    
    features_train = numeric_features.loc[train_idx]
    features_test = numeric_features.loc[test_idx]
    
    liquid_classes_train = liquid_classes.loc[train_idx]
    liquid_classes_test = liquid_classes.loc[test_idx]
    
    print(f"\n=== Train/Test Split ===")
    print(f"Training set: {len(train_idx)} rows")
    print(f"Test set: {len(test_idx)} rows")
    
    return (df_train, features_train, liquid_classes_train,
            df_test, features_test, liquid_classes_test)

def run_clustering(df, liquid_classes, n_clusters=5):
    X = df.drop(columns=["LiquidClassName"])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    results = pd.DataFrame({
        "LiquidClassName": liquid_classes,
        "Cluster": clusters
    })
    return results, X_pca, clusters

def train_autocomplete_model(features_train, features_test, holdout_fraction=0.2, random_state='random'):
    """Train model using pre-split train/test sets"""
    num_cols = features_train.shape[1]
    num_holdout = max(1, int(num_cols * holdout_fraction))
    
    # Randomly select columns to hold out
    if random_state == 'random':
        # Explicitly seed with current time to ensure different results each call
        import time
        np.random.seed(int(time.time() * 1000000) % (2**32))
    elif random_state is not None:
        np.random.seed(random_state)
    
    all_cols = features_train.columns.tolist()
    
    # NEVER holdout AsFlowRate - always keep it as input
    protected_cols = ['AsFlowRate']
    available_for_holdout = [col for col in all_cols if col not in protected_cols]
    
    # Shuffle to ensure randomness
    available_for_holdout = list(available_for_holdout)
    np.random.shuffle(available_for_holdout)
    
    holdout_cols = available_for_holdout[:num_holdout]
    given_cols = [col for col in all_cols if col not in holdout_cols]

    # Split features into input (X) and target (y) for both train and test
    X_train = features_train[given_cols]
    y_train = features_train[holdout_cols]
    
    X_test = features_test[given_cols]
    y_test = features_test[holdout_cols]

    print(f"\nModel configuration:")
    print(f"  Columns given as input: {len(given_cols)}")
    print(f"  Columns to predict: {len(holdout_cols)}")
    print(f"  Predicting: {', '.join(holdout_cols)}")

    model = xgb.XGBRegressor(n_estimators=100, random_state=42, max_depth=5)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Convert predictions to DataFrame
    y_pred_df = pd.DataFrame(y_pred, columns=holdout_cols, index=X_test.index)

    return X_test, y_test, y_pred_df

def print_autocomplete_examples(X_test, y_test, y_pred_df, liquid_classes_test, n_examples=5):
    print("\n=== Autocomplete Predictions Examples ===\n")
    
    # Randomly sample indices instead of always using the first ones
    sample_indices = np.random.choice(X_test.index, size=min(n_examples, len(X_test)), replace=False)
    
    for idx in sample_indices:
        liquid_name = liquid_classes_test.loc[idx]
        print(f"Liquid Class: {liquid_name} (Index: {idx})")
        print("Parameters given:")
        for col in X_test.columns:
            print(f"  {col}: {X_test.loc[idx, col]}")
        print("\nParameters predicted vs actual:")
        for col in y_test.columns:
            actual = y_test.loc[idx, col]
            predicted = y_pred_df.loc[idx, col]
            error = abs(predicted - actual)
            print(f"  {col}: Predicted={predicted:.3f}, Actual={actual:.3f}, Error={error:.3f}")
        print("\n" + "-"*40 + "\n")

def main(csv_path):
    # Step 1: Load and sanitize data (removes duplicates)
    df, numeric_features, liquid_classes = load_and_clean_data(csv_path)
    
    # Step 2: Split into train/test sets immediately
    (df_train, features_train, liquid_classes_train,
     df_test, features_test, liquid_classes_test) = split_train_test(
        df, numeric_features, liquid_classes, test_size=0.2
    )

    # Step 3: Run clustering on training data only
    cluster_results, X_pca, clusters = run_clustering(df_train, liquid_classes_train, n_clusters=5)
    print("\n=== Cluster Memberships (Training Set) ===")
    for cluster_id in sorted(cluster_results["Cluster"].unique()):
        members = cluster_results[cluster_results["Cluster"] == cluster_id]["LiquidClassName"].tolist()
        print(f"\nCluster {cluster_id} ({len(members)} members):")
        for m in members[:5]:  # Show first 5
            print(f"  - {m}")
        if len(members) > 5:
            print(f"  ... and {len(members) - 5} more")

    # Step 4: Train multiple models with different random holdout sets
    print("\n" + "="*60)
    print("AUTOCOMPLETE MODELS WITH DIFFERENT HOLDOUT SETS")
    print("="*60)
    
    for i in range(5):
        print(f"\n{'='*60}")
        print(f"MODEL {i+1}: Random holdout set")
        print('='*60)
        X_test, y_test, y_pred_df = train_autocomplete_model(
            features_train, 
            features_test,
            holdout_fraction=0.8,
            random_state='random'  # Use time-based randomness for different columns each time
        )
        print_autocomplete_examples(X_test, y_test, y_pred_df, liquid_classes_test, n_examples=10)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python model_comparison.py <path_to_csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    main(csv_path)