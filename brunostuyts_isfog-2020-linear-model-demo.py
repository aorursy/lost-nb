#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


data = pd.read_csv("/kaggle/input/training_data.csv")  # Store the contents of the csv file in the variable 'data'
data.head()


# In[4]:


data.describe()


# In[5]:


fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, sharey=True, figsize=(16,9))
ax1.scatter(data["qc [MPa]"], data["z [m]"], s=5)   # Create the cone tip resistance vs depth plot
ax2.scatter(data["Blowcount [Blows/m]"], data["z [m]"], s=5)  # Create the Blowcount vs depth plot 
ax3.scatter(data["Normalised ENTRHU [-]"], data["z [m]"], s=5) # Create the ENTHRU vs depth plot
# Format the axes (position, labels and ranges)
for ax in (ax1, ax2, ax3):
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.grid()
    ax.set_ylim(50, 0)
ax1.set_xlabel(r"Cone tip resistance, $ q_c $ (MPa)")
ax1.set_xlim(0, 120)
ax2.set_xlabel(r"Blowcount (Blows/m)")
ax2.set_xlim(0, 200)
ax3.set_xlabel(r"Normalised ENTRHU (-)")
ax3.set_xlim(0, 1)
ax1.set_ylabel(r"Depth below mudline, $z$ (m)")
# Show the plot
plt.show()


# In[6]:


# Select the data where the column 'Location ID' is equal to the location name
location_data = data[data["Location ID"] == "EK"]


# In[7]:


fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, sharey=True, figsize=(16,9))
# All data
ax1.scatter(data["qc [MPa]"], data["z [m]"], s=5)
ax2.scatter(data["Blowcount [Blows/m]"], data["z [m]"], s=5)
ax3.scatter(data["Normalised ENTRHU [-]"], data["z [m]"], s=5)
# Location-specific data
ax1.plot(location_data["qc [MPa]"], location_data["z [m]"], color='red')
ax2.plot(location_data["Blowcount [Blows/m]"], location_data["z [m]"], color='red')
ax3.plot(location_data["Normalised ENTRHU [-]"], location_data["z [m]"], color='red')
for ax in (ax1, ax2, ax3):
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.grid()
    ax.set_ylim(50, 0)
ax1.set_xlabel(r"Cone tip resistance (MPa)")
ax1.set_xlim(0, 120)
ax2.set_xlabel(r"Blowcount (Blows/m)")
ax2.set_xlim(0, 200)
ax3.set_xlabel(r"Normalised ENTRHU (-)")
ax3.set_xlim(0, 1)
ax1.set_ylabel(r"Depth below mudline (m)")
plt.show()


# In[8]:


fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(15,6))
# All data
ax1.scatter(data["qc [MPa]"], data["Blowcount [Blows/m]"], s=5)
ax2.scatter(data["Normalised ENTRHU [-]"], data["Blowcount [Blows/m]"], s=5)
ax3.scatter(data["z [m]"], data["Blowcount [Blows/m]"], s=5)
# Location-specific data
ax1.scatter(location_data["qc [MPa]"], location_data["Blowcount [Blows/m]"], color='red')
ax2.scatter(location_data["Normalised ENTRHU [-]"], location_data["Blowcount [Blows/m]"], color='red')
ax3.scatter(location_data["z [m]"], location_data["Blowcount [Blows/m]"], color='red')
for ax in (ax1, ax2, ax3):
    ax.grid()
    ax.set_ylim(0, 200)
    ax.set_ylabel(r"Blowcount (Blows/m)")
ax1.set_xlabel(r"Cone tip resistance (MPa)")
ax1.set_xlim(0, 120)
ax2.set_xlabel(r"Normalised ENTRHU (-)")
ax2.set_xlim(0, 1)
ax3.set_xlabel(r"Depth below mudline (m)")
ax3.set_xlim(0, 50)
plt.show()


# In[9]:


validation_ids = ['EL', 'CB', 'AV', 'BV', 'EF', 'DL', 'BM']
# Training data - ID not in validation_ids
training_data = data[~data['Location ID'].isin(validation_ids)]
# Validation data - ID in validation_ids
validation_data = data[data['Location ID'].isin(validation_ids)]


# In[10]:


features = ['Normalised ENTRHU [-]']
cleaned_training_data = training_data.dropna() # Remove NaN values
X = cleaned_training_data[features]
y = cleaned_training_data["Blowcount [Blows/m]"]


# In[11]:


from sklearn.linear_model import LinearRegression
model_1 = LinearRegression().fit(X,y)


# In[12]:


model_1.coef_, model_1.intercept_


# In[13]:


plt.scatter(X, y)
x = np.linspace(0.0, 1, 50)
plt.plot(x, model_1.intercept_ + model_1.coef_ * x, color='red')
plt.xlabel("Normalised ENTHRU (-)")
plt.ylabel("Blowcount (Blows/m)")
plt.show()


# In[14]:


model_1.score(X,y)


# In[15]:


plt.scatter(training_data["Normalised ENTRHU [-]"], training_data["Blowcount [Blows/m]"])
x = np.linspace(0, 1, 100)
plt.plot(x, 80 * np.tanh(5 * x - 0.5), color='red')
plt.xlabel("Normalised ENTHRU (-)")
plt.ylabel("Blowcount (Blows/m)")
plt.ylim([0.0, 175.0])
plt.show()


# In[16]:


Xlin = np.tanh(5 * cleaned_training_data[["Normalised ENTRHU [-]"]] - 0.5)


# In[17]:


plt.scatter(Xlin, y)
plt.xlabel(r"$ \tanh(5 \cdot ENTRHU_{norm} - 0.5) $")
plt.ylabel("Blowcount (Blows/m)")
plt.ylim([0.0, 175.0])
plt.show()


# In[18]:


model_2 = LinearRegression().fit(Xlin, y)


# In[19]:


model_2.coef_, model_2.intercept_


# In[20]:


plt.scatter(X, y)
x = np.linspace(0.0, 1, 50)
plt.plot(x, model_2.intercept_ + model_2.coef_ * (np.tanh(5*x - 0.5)), color='red')
plt.xlabel("Normalised ENTHRU (-)")
plt.ylabel("Blowcount (Blows/m)")
plt.ylim([0.0, 175])
plt.show()


# In[21]:


model_2.score(Xlin, y)


# In[22]:


enhanced_data = pd.DataFrame() # Create a dataframe for the data enhanced with the shaft friction feature
for location in training_data['Location ID'].unique(): # Loop over all unique locations
    locationdata = training_data[training_data['Location ID']==location].copy() # Select the location-specific data
    # Calculate the shaft resistance feature
    locationdata["Rs [kN]"] =         (np.pi * locationdata["Diameter [m]"] * locationdata["z [m]"].diff() * locationdata["qc [MPa]"]).cumsum()
    enhanced_data = pd.concat([enhanced_data, locationdata]) # Combine data for the different locations in 1 dataframe


# In[23]:


fig, ((ax1, ax2)) = plt.subplots(1, 2, sharey=True, figsize=(12,6))
ax1.scatter(enhanced_data["qc [MPa]"], enhanced_data["Blowcount [Blows/m]"])
ax2.scatter(enhanced_data["Rs [kN]"], enhanced_data["Blowcount [Blows/m]"])
x = np.linspace(0.0, 12000, 50)
ax2.plot(x, 85 * (np.tanh(0.001*x-1)), color='red')
ax1.set_xlabel("Cone tip resistance (MPa)")
ax2.set_xlabel("Shaft resistance (kN)")
ax1.set_ylabel("Blowcount (Blows/m)")
ax2.set_ylabel("Blowcount (Blows/m)")
ax1.set_ylim([0.0, 175])
plt.show()


# In[24]:


features = ["Rs [kN]"]
X = enhanced_data.dropna()[features]
y = enhanced_data.dropna()["Blowcount [Blows/m]"]
Xlin = np.tanh((0.001 * X) - 1)


# In[25]:


model_3 = LinearRegression().fit(Xlin, y)


# In[26]:


model_3.intercept_, model_3.coef_


# In[27]:


plt.scatter(X, y)
x = np.linspace(0.0, 12000, 50)
plt.plot(x, model_3.intercept_ + model_3.coef_ * (np.tanh(0.001*x - 1)), color='red')
plt.xlabel("Shaft resistance (kN)")
plt.ylabel("Blowcount (Blows/m)")
plt.ylim([0.0, 175])
plt.show()


# In[28]:


model_3.score(Xlin, y)


# In[29]:


plt.scatter(data["z [m]"], data["Blowcount [Blows/m]"])
z = np.linspace(0,35,100)
plt.plot(z, 100 * np.tanh(0.1 * z - 0.5), color='red')
plt.ylim([0, 175])
plt.xlabel("Depth (m)")
plt.ylabel("Blowcount (Blows/m)")
plt.show()


# In[30]:


enhanced_data["linearized ENTHRU"] = np.tanh(5 * enhanced_data["Normalised ENTRHU [-]"] - 0.5)
enhanced_data["linearized Rs"] = np.tanh(0.001 * enhanced_data["Rs [kN]"] - 1)
enhanced_data["linearized z"] = np.tanh(0.1 * enhanced_data["z [m]"] - 0.5)
linearized_features = ["linearized ENTHRU", "linearized Rs", "linearized z"]


# In[31]:


X = enhanced_data.dropna()[linearized_features]
y = enhanced_data.dropna()["Blowcount [Blows/m]"]
model_4 = LinearRegression().fit(X,y)


# In[32]:


model_4.score(X, y)


# In[33]:


model_4.intercept_, model_4.coef_


# In[34]:


predictions = model_4.predict(X)
predictions


# In[35]:


fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(15,6))
# Measurements
ax1.scatter(enhanced_data["Rs [kN]"], enhanced_data["Blowcount [Blows/m]"], s=5)
ax2.scatter(enhanced_data["Normalised ENTRHU [-]"], enhanced_data["Blowcount [Blows/m]"], s=5)
ax3.scatter(enhanced_data["z [m]"], enhanced_data["Blowcount [Blows/m]"], s=5)
# Predictions
ax1.scatter(enhanced_data.dropna()["Rs [kN]"], predictions, color='red')
ax2.scatter(enhanced_data.dropna()["Normalised ENTRHU [-]"], predictions, color='red')
ax3.scatter(enhanced_data.dropna()["z [m]"], predictions, color='red')
for ax in (ax1, ax2, ax3):
    ax.grid()
    ax.set_ylim(0, 175)
    ax.set_ylabel(r"Blowcount (Blows/m)")
ax1.set_xlabel(r"Shaft resistance (kN)")
ax1.set_xlim(0, 12000)
ax2.set_xlabel(r"Normalised ENTRHU (-)")
ax2.set_xlim(0, 1)
ax3.set_xlabel(r"Depth below mudline (m)")
ax3.set_xlim(0, 50)
plt.show()


# In[36]:


# Create a copy of the dataframe with location-specific data
validation_data_CB = validation_data[validation_data["Location ID"] == "CB"].copy()


# In[37]:


# Calculate the shaft resistance feature and put it in the column 'Rs [kN]'
validation_data_CB["Rs [kN]"] =     (np.pi * validation_data_CB["Diameter [m]"] *      validation_data_CB["z [m]"].diff() * validation_data_CB["qc [MPa]"]).cumsum()


# In[38]:


# Calculate linearized ENTHRU, Rs and z
validation_data_CB["linearized ENTHRU"] = np.tanh(5 * validation_data_CB["Normalised ENTRHU [-]"] - 0.5)
validation_data_CB["linearized Rs"] = np.tanh(0.001 * validation_data_CB["Rs [kN]"] - 1)
validation_data_CB["linearized z"] = np.tanh(0.1 * validation_data_CB["z [m]"] - 0.5)


# In[39]:


# Create the matrix with n samples and 3 features
X_validation = validation_data_CB.dropna()[linearized_features]
# Create the vector with n observations of blowcount
y_validation = validation_data_CB.dropna()["Blowcount [Blows/m]"]


# In[40]:


# Calculate the R2 score for the validation data
model_4.score(X_validation, y_validation)


# In[41]:


validation_predictions = model_4.predict(X_validation)


# In[42]:


fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(15,6))
# All data
ax1.plot(validation_data_CB["qc [MPa]"], validation_data_CB["z [m]"])
ax2.plot(validation_data_CB["Normalised ENTRHU [-]"], validation_data_CB["z [m]"])
ax3.plot(validation_data_CB["Blowcount [Blows/m]"], validation_data_CB["z [m]"])
# Location-specific data
ax3.scatter(validation_predictions, validation_data_CB.dropna()["z [m]"], color='red')
for ax in (ax1, ax2, ax3):
    ax.grid()
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_ylim(30, 0)
    ax.set_ylabel(r"Depth below mudline (m)")
ax1.set_xlabel(r"Cone tip resistance (MPa)")
ax1.set_xlim(0, 120)
ax2.set_xlabel(r"Normalised ENTRHU (-)")
ax2.set_xlim(0, 1)
ax3.set_xlabel(r"Blowcount (Blows/m)")
ax3.set_xlim(0, 175)
plt.show()


# In[43]:


final_data = pd.read_csv("/kaggle/input/validation_data.csv")
final_data.head()


# In[44]:


enhanced_final_data = pd.DataFrame() # Create a dataframe for the final data enhanced with the shaft friction feature
for location in final_data['Location ID'].unique(): # Loop over all unique locations
    locationdata = final_data[final_data['Location ID']==location].copy() # Select the location-specific data
    # Calculate the shaft resistance feature
    locationdata["Rs [kN]"] =         (np.pi * locationdata["Diameter [m]"] * locationdata["z [m]"].diff() * locationdata["qc [MPa]"]).cumsum()
    enhanced_final_data = pd.concat(
        [enhanced_final_data, locationdata]) # Combine data for the different locations in 1 dataframe


# In[45]:


enhanced_final_data.dropna(inplace=True) # Drop the rows containing NaN values and overwrite the dataframe


# In[46]:


enhanced_final_data["linearized ENTHRU"] = np.tanh(5 * enhanced_final_data["Normalised ENTRHU [-]"] - 0.5)
enhanced_final_data["linearized Rs"] = np.tanh(0.001 * enhanced_final_data["Rs [kN]"] - 1)
enhanced_final_data["linearized z"] = np.tanh(0.1 * enhanced_final_data["z [m]"] - 0.5)


# In[47]:


# Create the matrix with n samples and 3 features
X = enhanced_final_data[linearized_features]


# In[48]:


final_predictions = model_4.predict(X)


# In[49]:


enhanced_final_data["Blowcount [Blows/m]"] = final_predictions


# In[50]:


enhanced_final_data[["ID", "Blowcount [Blows/m]"]].to_csv("sample_submission_linearmodel.csv", index=False)

