import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, confusion_matrix, classification_report

warnings.filterwarnings('ignore')

# Load the datasets
imdb_top_1000 = pd.read_csv('C:/Users/gudde/OneDrive/Desktop/ml_glob/datasets/imdb_top_1000.csv')
movies_data = pd.read_csv('C:/Users/gudde/OneDrive/Desktop/ml_glob/datasets/movies_data.csv')

# Display the first few rows of each dataset
print(imdb_top_1000.head(), movies_data.head())

# Check for missing values and data types
print(imdb_top_1000.info(), movies_data.info())

# Convert 'Released_Year' to integer
imdb_top_1000['Released_Year'] = imdb_top_1000['Released_Year'].str.extract('(\d+)').astype(float)
movies_data['Released_Year'] = movies_data['Released_Year'].str.extract('(\d+)').astype(float)

# Convert 'Runtime' to integer
imdb_top_1000['Runtime'] = imdb_top_1000['Runtime'].str.extract('(\d+)').astype(float)

# Convert 'Gross' to numeric, removing commas
imdb_top_1000['Gross'] = imdb_top_1000['Gross'].str.replace(',', '').astype(float)
movies_data['Gross'] = movies_data['Gross'].astype(float)

# Fill missing values in 'Meta_score' with the mean
imdb_top_1000['Meta_score'].fillna(imdb_top_1000['Meta_score'].mean(), inplace=True)
movies_data['Meta_score'].fillna(movies_data['Meta_score'].mean(), inplace=True)

# Check the cleaned data
print(imdb_top_1000.head(), movies_data.head())

# Plot the distribution of genres
plt.figure(figsize=(12, 6))
sns.countplot(y='Genre', data=imdb_top_1000, order=imdb_top_1000['Genre'].value_counts().index)
plt.title('Distribution of Movie Genres')
plt.xlabel('Count')
plt.ylabel('Genre')
plt.show()

# Top 10 directors
top_directors = imdb_top_1000['Director'].value_counts().head(10)
plt.figure(figsize=(10, 5))
sns.barplot(x=top_directors.values, y=top_directors.index)
plt.title('Top 10 Directors')
plt.xlabel('Number of Movies')
plt.ylabel('Director')
plt.show()

# Scatter plot of IMDB Rating vs. Number of Votes
plt.figure(figsize=(10, 6))
sns.scatterplot(x='IMDB_Rating', y='No_of_Votes', data=imdb_top_1000)
plt.title('IMDB Rating vs. Number of Votes')
plt.xlabel('IMDB Rating')
plt.ylabel('Number of Votes')
plt.show()

# Prepare the data
features = ['Released_Year', 'Runtime', 'Meta_score', 'No_of_Votes']
X = imdb_top_1000[features].fillna(0)
y = imdb_top_1000['IMDB_Rating']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2 Score): {r2:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")

# Add a confusion matrix and classification report
# Convert ratings into categories (e.g., Good/Bad based on threshold)
threshold = 7.0
y_test_binary = (y_test >= threshold).astype(int)  # Good (>=7) or Bad (<7)
y_pred_binary = (y_pred >= threshold).astype(int)

conf_matrix = confusion_matrix(y_test_binary, y_pred_binary)

# Check if both classes are present
if len(np.unique(y_test_binary)) > 1 and len(np.unique(y_pred_binary)) > 1:
    report = classification_report(y_test_binary, y_pred_binary, target_names=['Bad', 'Good'])
    print("\nClassification Report:\n", report)
else:
    print("\nWarning: Only one class present in the predictions. Classification report cannot be generated.")

print("\nConfusion Matrix:\n", conf_matrix)



# Scatter plot of actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.title("Actual vs Predicted IMDb Ratings")
plt.show()

# Add a feature to predict rating based on user input
def predict_movie_rating():
    print("\nEnter the details of the movie:")
    try:
        year = float(input("Released Year: "))
        runtime = float(input("Runtime (in minutes): "))
        meta_score = float(input("Meta Score (0-100): "))
        no_of_votes = float(input("Number of Votes: "))

        # Create a single-row DataFrame for the input
        input_data = pd.DataFrame([[year, runtime, meta_score, no_of_votes]], columns=features)
        prediction = model.predict(input_data)
        print(f"\nPredicted IMDb Rating: {prediction[0]:.2f}")
    except ValueError:
        print("Invalid input. Please enter numeric values.")

# Call the function
predict_movie_rating()
