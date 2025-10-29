# Naive Bayes Sentiment Classifier with Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Sample dataset
data = [
    ("I love this movie!", "positive"),
    ("This film is amazing", "positive"),
    ("I enjoyed the story", "positive"),
    ("This movie is boring", "negative"),
    ("I hate this film", "negative"),
    ("Worst movie ever", "negative"),
]

# Split features and labels
texts, labels = zip(*data)

# Convert text into numerical form
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

# Create and train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"])
plt.title("Naive Bayes Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Accuracy Bar Visualization
plt.bar(["Naive Bayes Accuracy"], [acc], color='green')
plt.ylim(0, 1)
plt.title("Model Accuracy")
plt.ylabel("Accuracy Score")
plt.show()

# Test your own text
sample = ["I really love this movie", "I dislike the acting"]
sample_vec = vectorizer.transform(sample)
print("\nSample Predictions:", model.predict(sample_vec))
