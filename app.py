#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# # **Generic Machine Learning Pipeline Outline**
# 
# ## **1. Load the Dataset**
# - Import necessary libraries  
# - Read the dataset from a local file or an online source  
# - Display basic dataset information (`head()`, `describe()`, `info()`)  
# 
# ## **2. Exploratory Data Analysis (EDA)**
# - Check for missing values  
# - Visualize feature distributions (e.g., histograms, boxplots)  
# - Analyze relationships between features  
# 
# ## **3. Data Preprocessing**
# - Handle missing values (e.g., imputation or removal)  
# - Convert categorical variables to numerical (e.g., label encoding, one-hot encoding)  
# - Feature scaling (e.g., StandardScaler, MinMaxScaler)  
# 
# ## **4. Split Data into Training and Testing Sets**
# - Use `train_test_split()` to divide data into training and test sets  
# - Set a `random_state` for reproducibility  
# 
# ## **5. Model Selection & Training**
# - Choose an appropriate ML model (e.g., Linear Regression, Decision Tree, SVM, etc.)  
# - Fit the model on the training data  
# 
# ## **6. Model Evaluation**
# - Make predictions on the test set  
# - Compute evaluation metrics:  
#   - **For Regression:** Mean Squared Error (MSE), R² Score  
#   - **For Classification:** Accuracy, Precision, Recall, F1-score, Confusion Matrix  
# 
# ## **7. Model Saving & Deployment**
# - Save the trained model using `pickle` or `joblib`  
# - Load the saved model and test predictions on new data  
# 
# ## **8. Model Optimization (Optional)**
# - Hyperparameter tuning (e.g., GridSearchCV, RandomizedSearchCV)  
# - Feature selection techniques  
# 

# 

# # **Generic Machine Learning Pipeline Outline**

# ## **1. Load the Dataset**
# - Import necessary libraries  
# - Read the dataset from a local file or an online source  
# - Display basic dataset information (`head()`, `describe()`, `info()`)  

# In[2]:


#get_ipython().system('pip install streamlit')


# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

import pickle


# In[22]:


df = pd.read_csv("Iris.csv")
df


# In[23]:


df.describe()


# ## **2. Exploratory Data Analysis (EDA)**
# - Check for missing values  
# - Visualize feature distributions (e.g., histograms, boxplots)  
# - Analyze relationships between features  

# In[24]:


df.info()


# In[25]:


df.isna().sum()


# ## **3. Data Preprocessing**
# - Handle missing values (e.g., imputation or removal)  
# - Convert categorical variables to numerical (e.g., label encoding, one-hot encoding)  
# - Feature scaling (e.g., StandardScaler, MinMaxScaler)  
# 

# In[26]:


X = df.drop(columns=["Species", "Id"])
Y = df["Species"]
# X is the feature and Y is the target


# In[27]:


Y.unique()


# In[28]:


Y = LabelEncoder().fit_transform(Y)
Y


# In[29]:


X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.3, random_state=82)


# In[30]:


X_train


# In[31]:


model = LinearRegression()
model.fit(X_train, y_train)


# In[32]:


y_pred = model.predict(X_test)


# In[33]:


y_pred


# In[34]:


y_pred_class = np.round(y_pred).astype(int)
# Rounds y_pred to nearest integer
#It is necessary to round off, because the targets are originally labels


# In[35]:


y_pred_class


# In[36]:


y_test


# In[38]:


y_pred_class == y_test


# In[39]:


#MSE
#R2-SCORE
#ACCURACY
print("Mean Square Error", mean_squared_error(y_pred, y_test))
print("R2_SCORE", r2_score(y_pred, y_test))
print("Accuracy", accuracy_score(y_pred_class, y_test))


# In[40]:


with open("first_iris_model.pkl", "wb") as file:
    pickle.dump(model, file)


# In[41]:


with open("first_iris_model.pkl", "rb") as file:
    loaded_model = pickle.load(file)


# In[42]:


sample = np.array([[5.1, 3.5, 1.4, 0.2]])
prediction = loaded_model.predict(sample)


# In[43]:


prediction


# In[44]:


prediction.round().astype(int)


# In[45]:


df.loc[5, "Species"]


# In[48]:


df.iloc[3, 5]


# In[51]:


# Load the model
with open("first_iris_model.pkl", 'rb') as file:
    model= pickle.load(file)


# In[52]:


# Streamlit UI

st.title('Iris Flower Prediction App')
st.write('This app predicts the **Iris Flower** type!')
st.write('Please input the followimg parameters:')


# In[53]:


# Input form
sepal_ID = st.number_input('Sepal ID', min_value=0.1, max_value=10.0, value=5.4, step=0.1)
sepal_width = st.number_input('Sepal Width', min_value=0.1, max_value=10.0, value=3.4, step=0.1)
petal_length = st.number_input('Petal Length', min_value=0.1, max_value=10.0, value=1.3, step=0.1)
sepal_length = st.number_input('Sepal Length', min_value=0.1, max_value=10.0, value=0.2, step=0.1)
petal_width = st.number_input('Pepal Width', min_value=0.1, max_value=10.0, value=0.2, step=0.1)


# In[54]:


# Prediction

if st.button('Predict'):
    user_input = np.array([sepal_ID, sepal_width, petal_length, sepal_length, petal_width])
    prediction = model.predic(user_input)
    species_mapping = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    st.write(prediction)
    predicted_species = species_mapping.get(int(prediction[0]), 'unknown')
    st.write(f'The predicted species is: {predicted_species}')


# In[55]:


#footer
st.write("Made with streamlit")

