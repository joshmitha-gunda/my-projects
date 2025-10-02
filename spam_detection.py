import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from textblob import TextBlob
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('/content/drive/MyDrive/Youtube-Spam-Dataset.csv')
# Display the first few rows of the dataset
print("Columns in the dataset:", data.columns)
print("First few rows of the dataset:\n", data.head())

# Alternative Preprocessing function without NLTK
def preprocess_text(text):
    # Convert to lowercase and remove non-alphanumeric characters
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    # Remove stopwords
    tokens = [word for word in text.split() if word not in ENGLISH_STOP_WORDS]
    return ' '.join(tokens)

# Apply preprocessing to the CONTENT column
data['CONTENT'] = data['CONTENT'].apply(preprocess_text)

# Add a new feature: sentiment score
data['SENTIMENT'] = data['CONTENT'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Train-Test Split
X = data[['CONTENT', 'SENTIMENT']]
y = data['CLASS']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train['CONTENT'])
X_test_tfidf = tfidf_vectorizer.transform(X_test['CONTENT'])

# Combine TF-IDF and SENTIMENT features
X_train_combined = np.hstack((X_train_tfidf.toarray(), X_train['SENTIMENT'].values.reshape(-1, 1)))
X_test_combined = np.hstack((X_test_tfidf.toarray(), X_test['SENTIMENT'].values.reshape(-1, 1)))

# Initialize models for stacking
base_models = [
    ('gnb', GaussianNB()),
    ('rf', RandomForestClassifier(random_state=42)),
]

# Ensemble model with stacking
stacking_clf = StackingClassifier(estimators=base_models, final_estimator=LogisticRegression())

# Train the Stacked Model
stacking_clf.fit(X_train_combined, y_train)

# Predictions and Evaluation
y_pred = stacking_clf.predict(X_test_combined)
print(f"Stacked Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Plotting Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Spam', 'Spam'], yticklabels=['Non-Spam', 'Spam'])
plt.title('Confusion Matrix for Stacked Model')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# SHAP Explainability
# Assuming multi-class classification, select SHAP values for the class of interest
explainer = shap.Explainer(stacking_clf.named_estimators_['rf'], X_train_combined)
shap_values = explainer(X_test_combined)

# For multi-class, shap_values will have a shape of (num_samples, num_features, num_classes)
# Let's plot SHAP values for each class separately
for class_idx in range(shap_values.shape[-1]):  # Iterate over each class
    shap_values_class = shap_values[:, :, class_idx]

    # Plot the SHAP values for the current class
    print(f"Plotting SHAP values for class {class_idx}")
    shap.summary_plot(shap_values_class, X_test_combined, feature_names=tfidf_vectorizer.get_feature_names_out().tolist() + ['SENTIMENT'])