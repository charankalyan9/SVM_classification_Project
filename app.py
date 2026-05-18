import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Iris SVM Regression", layout="centered")

st.title("🌸 Iris Dataset - SVM Regression")
st.write("This Streamlit app performs Support Vector Machine (SVR) regression using the Iris dataset.")

# Load dataset
iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

# Using target as regression output
df["target"] = iris.target

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Features and target
X = df.drop("target", axis=1)
y = df["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Sidebar controls
st.sidebar.header("SVR Parameters")

C = st.sidebar.slider("C", 0.1, 100.0, 1.0)
epsilon = st.sidebar.slider("Epsilon", 0.01, 1.0, 0.1)
kernel = st.sidebar.selectbox(
    "Kernel",
    ("rbf", "linear", "poly", "sigmoid")
)

# Train model
model = SVR(C=C, epsilon=epsilon, kernel=kernel)
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("Model Performance")
st.write(f"**Mean Squared Error:** {mse:.4f}")
st.write(f"**R² Score:** {r2:.4f}")

# Results table
results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

st.subheader("Prediction Results")
st.dataframe(results)

st.success("SVM Regression completed successfully!")