from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate a synthetic dataset
X, y = make_classification(
    n_samples=1000, n_features=4, n_informative=2, n_redundant=0, random_state=42
)

# Split dataset into training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Initialize the Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the model on training data
rf.fit(X_train, y_train)

# Predict on testing data
predictions = rf.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, predictions)

accuracy
