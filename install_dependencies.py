import os

# List of required packages
packages = [
    "pandas", "numpy", "matplotlib", "seaborn", "nest_asyncio",
    "googletrans==4.0.0-rc1", "nltk", "wordcloud", "imbalanced-learn",
    "category_encoders", "scikit-learn", "mlxtend", "xgboost", "hyperopt", "geopy","asyncio","regex","shap"
]

# Install each package
for package in packages:
    os.system(f"pip install {package}")

print("✅ All dependencies installed successfully!")
