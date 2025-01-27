# Dataset imported from https://www.kaggle.com/datasets/hamjashaikh/mental-health-detection-dataset
# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import  accuracy_score
import matplotlib.pyplot as plt


# Read the dataset
df = pd.read_csv('mental_health_detection_dataset.csv')

# Statistical analysis
print("Descriptive Statistics:")
print(df.info())
print(df.describe())

# Missing data analysis
print("\n Missing data analysis:")
print(df.isnull().sum())

# Drop missing datas
df = df.dropna()

# Correlation analysis
numeric_df = df.drop(['Depression State', 'Number '], axis=1)
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Correlation matrix between symptoms')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.close()



# Create variables for the Decision Tree Classifier
X = df.drop(['Depression State', 'Number '], axis=1)
y = df['Depression State']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the model and train it
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Make predictions
y_pred = dt_model.predict(X_test)

# Evaluate the model performance
print("\nModel Performance Report:")
print("\nPrediction Results:")
print("=" * 50)
print("Actual State vs Model Prediction")
print("=" * 50)

# Show the first 10 examples
for real, pred in zip(y_test[:10], y_pred[:10]):
    print(f"Actual State: {real:15} | Prediction: {pred}")

print("\nModel Accuracy Rate: {:.1f}%".format(accuracy_score(y_test, y_pred)*100))






