import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def load_and_clean_data(csv_path):
    # Load CSV
    df = pd.read_csv(csv_path)

    # Assume the liquid class names are in a column called "LiquidClassName"
    # If your CSV has a different name, adjust here
    if "LiquidClassName" not in df.columns:
        # Try to guess: first column is probably the name
        df.rename(columns={df.columns[0]: "LiquidClassName"}, inplace=True)

    # Separate names from features
    liquid_classes = df["LiquidClassName"]
    features = df.drop(columns=["LiquidClassName"])

    # Only fill missing values for numeric columns
    numeric_features = features.select_dtypes(include=["number"])
    non_numeric_features = features.drop(columns=numeric_features.columns, errors="ignore")

    numeric_features = numeric_features.fillna(numeric_features.median())

    # Recombine
    clean_df = pd.concat([liquid_classes, numeric_features, non_numeric_features], axis=1)

    return clean_df, liquid_classes

def run_clustering(df, liquid_classes, n_clusters=5):
    # Drop name column before clustering
    X = df.drop(columns=["LiquidClassName"])

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA (2D for visualization, but mainly for dimension reduction)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    # Add results back
    results = pd.DataFrame({
        "LiquidClassName": liquid_classes,
        "Cluster": clusters
    })

    return results, X_pca, clusters

def main(csv_path):
    df, liquid_classes = load_and_clean_data(csv_path)
    results, X_pca, clusters = run_clustering(df, liquid_classes, n_clusters=5)

    # Print cluster memberships
    for cluster_id in sorted(results["Cluster"].unique()):
        members = results[results["Cluster"] == cluster_id]["LiquidClassName"].tolist()
        print(f"\nCluster {cluster_id} ({len(members)} members):")
        for m in members:
            print(f"  - {m}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python model_comparison.py <path_to_csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    main(csv_path)
