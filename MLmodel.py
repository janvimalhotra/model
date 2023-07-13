import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
from xgboost import XGBClassifier
from scipy.stats import zscore

# Load the dataset
df = pd.read_csv('customer_churn.csv')

# Extract locality information
df['Locality'] = df['Location'].str.replace(r'\d+', '').str.split(',').str[0].str.strip()

# Encoding locality information
label_encoder = LabelEncoder()
df['Locality_Encoded'] = label_encoder.fit_transform(df['Locality'])

# drop 'Location', 'Locality','Names'
df.drop(['Location', 'Locality', 'Names'], axis=1, inplace=True)

# Extract company information
company_encoder = LabelEncoder()

# Encode company information
df['Company_Encoded'] = company_encoder.fit_transform(df['Company'])

# drop 'Onboard_date', 'Company'
df.drop(['Onboard_date', 'Company'], axis=1, inplace=True)

# Outlier removal
z_scores = zscore(df)
abs_z_scores = abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
df = df[filtered_entries]

# Feature Engineering
# Calculate Purchase_Per_site
df['Purchase_Per_Site'] = df['Total_Purchase'] / df['Num_Sites']
# Calculate Purchase_Per_Year
df['Purchase_Per_Year'] = df['Total_Purchase'] / df['Years']

# Make X, y
X = df.drop('Churn', axis=1)
y = df['Churn']

# Assign feature names to encoded dataset
X.columns = ['Age', 'Total_Purchase', 'Account_Manager', 'Years', 'Num_Sites', 'Locality_Encoded', 'company_encoded', 'Purchase_Per_Site',
       'Purchase_Per_Year']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost Classifier
xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)

pickle.dump(xgb_model,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))