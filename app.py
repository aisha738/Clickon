# Streamlit UI
import streamlit as st
import numpy as np
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
    prediction = model.predict(user_input)
    species_mapping = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    st.write(prediction)
    predicted_species = species_mapping.get(int(prediction[0]), 'unknown')
    st.write(f'The predicted species is: {predicted_species}')


# In[55]:


#footer
st.write("Made with streamlit")

