# Install necessary libraries
# pip install pandas scikit-learn

# Import libraries
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load candidate information data
candidate_info = pd.read_csv('candidates_info.csv')  # Replace with your candidate info file

# Load social media data
social_media_data = pd.read_csv('socialmedia.csv')  # Replace with your social media data file

# Merge the two datasets based on 'User ID'
merged_data = pd.merge(social_media_data, candidate_info, left_on='UserID', right_on='UserID', how='inner')

# Assuming 'label' is the target variable indicating candidate support
# Assuming 'Candidate_Support' is the column indicating support for a candidate
X = merged_data[['PostText', 'UserBio', 'UserDescription1', 'UserDescription2']]
y = merged_data['Candidate_Support']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text data to numerical features
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train['PostText'].astype(str))
X_test_vectorized = vectorizer.transform(X_test['PostText'].astype(str))

# Check the shapes
print("X_train shape:", X_train_vectorized.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test_vectorized.shape)
print("y_test shape:", y_test.shape)

# Choose a machine learning algorithm
model = MultinomialNB()

# Train the model
model.fit(X_train_vectorized, y_train)


# Make predictions on the test set
y_pred = model.predict(X_test_vectorized)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('\nConfusion Matrix:')
print(conf_matrix)
print('\nClassification Report:')
print(classification_rep)
