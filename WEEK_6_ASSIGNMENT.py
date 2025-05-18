import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import numpy as np

# --- Task 1: Load and Explore the Dataset ---

print("--- Task 1: Load and Explore the Dataset ---")

try:
    # Load the Iris dataset
    iris = load_iris()
    iris_df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                           columns=iris['feature_names'] + ['species'])

    # Map numerical species to names for better readability
    species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    iris_df['species'] = iris_df['species'].map(species_map)

    print("\nFirst 5 rows of the dataset:")
    print(iris_df.head())

    print("\nDataset Info (data types and non-null counts):")
    iris_df.info()

    print("\nChecking for missing values:")
    print(iris_df.isnull().sum())

    # Cleaning the dataset (demonstrating the concept, Iris dataset is clean)
    if iris_df.isnull().sum().any():
        print("\nMissing values found. Attempting to fill or drop...")
        # For demonstration, let's say we choose to drop rows with any missing values
        # iris_df.dropna(inplace=True)
        # Or fill with a common strategy (e.g., mean for numerical columns)
        # iris_df.fillna(iris_df.mean(numeric_only=True), inplace=True)
        print("No missing values found in Iris dataset. No cleaning action needed.")
    else:
        print("\nNo missing values found. Dataset is clean.")

except FileNotFoundError:
    print("Error: Dataset file not found. Please ensure the path is correct.")
except Exception as e:
    print(f"An unexpected error occurred during dataset loading or exploration: {e}")


# --- Task 2: Basic Data Analysis ---

print("\n--- Task 2: Basic Data Analysis ---")

print("\nBasic statistics of numerical columns:")
print(iris_df.describe())

print("\nMean of numerical columns grouped by species:")
# Ensure species is treated as a categorical column for grouping
species_group_mean = iris_df.groupby('species')[iris.feature_names].mean()
print(species_group_mean)

print("\n--- Findings from Basic Data Analysis ---")
print("1. Setosa has generally smaller sepal and petal dimensions compared to Versicolor and Virginica.")
print("2. Virginica tends to have the largest sepal and petal dimensions.")
print("3. There's a clear differentiation in petal length and width across the species, which makes them good features for classification.")
print("4. Sepal width for setosa is on average larger than for versicolor and virginica, which is an interesting counter-trend to other dimensions.")


# --- Task 3: Data Visualization ---

print("\n--- Task 3: Data Visualization ---")

sns.set_style("whitegrid") # Apply seaborn style

# 1. Line chart showing trends over time (simulated)
# We'll sort by petal length and plot sepal length to simulate a trend,
# as Iris doesn't have a natural time series.
plt.figure(figsize=(10, 6))
iris_df_sorted = iris_df.sort_values(by='petal length (cm)').reset_index(drop=True)
plt.plot(iris_df_sorted.index, iris_df_sorted['sepal length (cm)'], color='purple', linestyle='-', marker='o', markersize=4)
plt.title('Simulated Trend: Sepal Length Sorted by Petal Length', fontsize=16)
plt.xlabel('Sorted Index (by Petal Length)', fontsize=12)
plt.ylabel('Sepal Length (cm)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
print("Insight: This simulated line chart shows how sepal length varies when ordered by petal length. It demonstrates how a line chart can display trends, even if the 'trend' here is not temporal but based on another feature's order.")


# 2. Bar chart showing the comparison of a numerical value across categories
plt.figure(figsize=(8, 6))
sns.barplot(x='species', y='petal length (cm)', data=iris_df, palette='viridis')
plt.title('Average Petal Length per Species', fontsize=16)
plt.xlabel('Species', fontsize=12)
plt.ylabel('Average Petal Length (cm)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
print("Insight: This bar chart clearly shows that Setosa has the shortest average petal length, followed by Versicolor, and then Virginica with the longest petals. This strong separation is key for species identification.")


# 3. Histogram of a numerical column to understand its distribution
plt.figure(figsize=(8, 6))
sns.histplot(iris_df['sepal width (cm)'], kde=True, bins=15, color='teal')
plt.title('Distribution of Sepal Width (cm)', fontsize=16)
plt.xlabel('Sepal Width (cm)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.tight_layout()
plt.show()
print("Insight: The histogram of sepal width shows a somewhat normal distribution, with a peak around 3.0-3.5 cm. There's a slight skew towards the higher end, indicating some species might have wider sepals.")


# 4. Scatter plot to visualize the relationship between two numerical columns
plt.figure(figsize=(10, 7))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=iris_df, s=80, alpha=0.8, palette='Set1')
plt.title('Relationship between Sepal Length and Petal Length by Species', fontsize=16)
plt.xlabel('Sepal Length (cm)', fontsize=12)
plt.ylabel('Petal Length (cm)', fontsize=12)
plt.legend(title='Species')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
print("Insight: This scatter plot is highly informative. It clearly separates the Setosa species based on both sepal and petal length (small values). Versicolor and Virginica show some overlap but are generally separable, with Virginica having larger sepal and petal lengths. This plot highlights the strong linear relationship between these two features within each species group.")
