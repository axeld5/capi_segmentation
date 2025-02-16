import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib  # for saving the model

# Load the saved features and masks
features = torch.load("full_features.pt")  # Shape: (N, feature_dim)
masks = torch.load("full_masks.pt")          # Shape: (N, mask_dim)

# Convert tensors to NumPy arrays (scikit-learn requires NumPy arrays)
X = features.numpy()
y = masks.numpy()

# Convert mask values to integers (assuming binary masks 0 or 1)
y = y.astype(np.int64)

print("Features shape:", X.shape)
print("Masks shape:", y.shape)

# Optionally, split into train/test sets for evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create a logistic regression model wrapped for multi-output classification.
# This trains one logistic regression classifier per output (pixel).
base_lr = LogisticRegression(solver='lbfgs', max_iter=1000)
model = MultiOutputClassifier(base_lr, n_jobs=-1)

# Train the logistic regression model
print("Training logistic regression...")
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Compute pixel-level accuracy
accuracy = np.mean(y_pred == y_test)
print(f"Test pixel-level accuracy: {accuracy:.4f}")

# Save the trained logistic regression model to disk
joblib.dump(model, 'logreg_model.pkl')
print("Model saved to logreg_model.pkl")
