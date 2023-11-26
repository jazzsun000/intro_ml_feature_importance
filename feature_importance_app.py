import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import shap

st.title("Feature Importance Demonstrator")

st.write("""
    Welcome to the Feature Importance Demonstrator. This app showcases different methods of evaluating feature importance 
    in a regression model. Choose a demonstration from the sidebar to see how feature importance is calculated and visualized.
""")

# Function to plot regression and calculate p-value
def plot_regression_and_p_value(X, y):
    if X.ndim == 1:
        X = X[:, np.newaxis]
    slope, intercept, r_value, p_value, std_err = stats.linregress(X[:, 0], y)
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], y, label="Data")
    ax.plot(X[:, 0], intercept + slope*X[:, 0], 'r', label=f"Fitted line (p-value={p_value:.5f})")
    ax.set_xlabel('Feature Value')
    ax.set_ylabel('Target')
    ax.legend()
    return p_value, fig

# Function to plot permutation importance
def plot_permutation_importance(model, X, y):
    result = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=2)
    sorted_idx = result.importances_mean.argsort()
    fig, ax = plt.subplots()
    ax.boxplot(result.importances[sorted_idx].T, vert=False, labels=np.array(range(X.shape[1]))[sorted_idx])
    ax.set_title("Permutation Importance of Features")
    ax.set_xlabel("Decrease in model performance")
    ax.set_ylabel("Features (by index)")
    return fig

# Function to plot SHAP values
def plot_shap_values(model, X):
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    fig = plt.figure()
    shap.summary_plot(shap_values, X, plot_type="bar")
    return fig

# Sidebar setup
st.sidebar.header("Feature Importance Demonstrations")
option = st.sidebar.selectbox("Choose the demonstration", 
                              ["Unreliability of P-Values", "Permutation Importance", "Tree Explainers"])

# Generate a synthetic dataset
X, y = make_regression(n_samples=100, n_features=20, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Create a random forest regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Show the selected demonstration
if option == "Unreliability of P-Values":
    st.sidebar.subheader("Unreliability of P-Values")
    num_samples = st.sidebar.slider("Number of resamples", 5, 100, 20)
    resample_size = st.sidebar.slider("Resample Size (as % of total)", 10, 100, 10)

    st.write("""
        **Unreliability of P-Values**  
        This demonstration shows how the p-value of a feature can vary with different subsets of the data. 
        A low p-value suggests that the feature is significantly associated with the target variable.
    """)

    original_p_value, original_fig = plot_regression_and_p_value(X[:, [0]], y)
    st.pyplot(original_fig)
    st.write(f"Original P-value: {original_p_value:.5f}")

    # Resample and calculate p-values
    p_values = []
    for _ in range(num_samples):
        resampled_indices = np.random.choice(range(X.shape[0]), size=int(resample_size / 100 * X.shape[0]), replace=False)
        X_resampled = X[resampled_indices]
        y_resampled = y[resampled_indices]
        p_value, _ = plot_regression_and_p_value(X_resampled, y_resampled)
        p_values.append(p_value)

    # Plot the distribution of p-values
    fig, ax = plt.subplots()
    ax.hist(p_values, bins=10, color='skyblue', edgecolor='black')
    ax.set_title('Distribution of P-Values from Resampling')
    ax.set_xlabel('P-Value')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

elif option == "Permutation Importance":
    st.write("""
        **Permutation Importance**  
        This method evaluates the importance of a feature by calculating the increase in the model's prediction error 
        after permuting the feature. A higher value means the feature is more important for model predictions.
    """)
    
    fig = plot_permutation_importance(model, X_test, y_test)
    st.pyplot(fig)

elif option == "Tree Explainers":
    st.write("""
        **Tree Explainers (SHAP Values)**  
        SHAP values explain the contribution of each feature to the prediction of an individual instance. 
        In this plot, each bar represents the average impact of a feature on the model output.
    """)
    
    fig = plot_shap_values(model, X_test)
    st.pyplot(fig)
