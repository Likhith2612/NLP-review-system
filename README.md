# Sentiment Analysis on Restaurant Reviews

## Overview  
This project focuses on analyzing restaurant reviews to determine whether they are positive or negative. Using machine learning techniques, we classify text reviews based on sentiment. The dataset contains 1000 reviews labeled as either liked (1) or not liked (0).  

## Dataset  
The dataset (`Restaurant_Reviews.csv`) consists of two columns:  
- **Review**: The text of the restaurant review.  
- **Liked**: A binary label indicating positive (1) or negative (0) sentiment.  

## Libraries Used  
- pandas  
- matplotlib  
- sklearn (for text vectorization and model training)  

## Steps Involved  

### 1. Data Exploration  
- Loaded the dataset using `pandas` and checked its structure.  
- Plotted basic visualizations to understand sentiment distribution.  

### 2. Data Preprocessing  
- Extracted the `Review` column as input (X) and `Liked` as the target variable (y).  
- Split the dataset into training (80%) and testing (20%).  
- Applied **TF-IDF Vectorization** to convert text into numerical form.  

### 3. Model Training  
Two machine learning models were used for classification:  
- **Logistic Regression**: A linear model for binary classification.  
- **Multinomial Naïve Bayes**: A probabilistic classifier suitable for text data.  

### 4. Model Evaluation  
- Trained both models on the vectorized review data.  
- Evaluated model accuracy using the test dataset.  
- The final accuracy achieved was **79.5%**.  

### 5. Predicting New Reviews  
- The trained model can classify new reviews based on sentiment.  
- Example prediction:  
  ```python
  new_text_data = ["Food is bad"]  
  predictions = model.predict(tfidf_vectorizer.transform(new_text_data))  
  print(predictions)  # Output: [0] (negative review)
