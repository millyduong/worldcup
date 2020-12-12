import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Write text header
st.header('NEW YORK CITY AIRBNB')

# Load data
@st.cache
def load_data(path):
    data = pd.read_csv('/Users/millyduong/Desktop/demo4/nyc_airbnb.csv')
    return data

df = load_data('/Users/nhanpham/CoderSchool/streamlit_demo/nyc_airbnb.csv')
st.dataframe(df)

# Display a dataframe
st.subheader('Price per Neighborhood Group')
price_neigh = df.groupby('neighbourhood_group').mean()[['price']]
st.dataframe(price_neigh)

# Plot price per neighborhoud with Matplotlib
plot_index = price_neigh.index.to_list()
plot_value = price_neigh['price'].to_list()
plt.figure(figsize=(5, 5))
plt.bar(plot_index, plot_value)
st.pyplot()


st.subheader('Price Range')
# Plot Geomap with longitude and latitude
slide = st.slider("Choose minimum price", 1, 1000, 1)

entire_bnb = df[(df['room_type'] == 'Entire home/apt') &
                (df['price'] > int(slide))][['latitude', 'longitude']]
# st.dataframe(entire_bnb)
st.map(entire_bnb)


st.subheader('Price by neighborhood')
# Plot comparison between house price based on neighborhood group

# Drop-down list
option = st.selectbox('Select an area', ('Brooklyn', 'Manhattan', 'Queens', 'Bronx', 'Staten Island'))

# Price per type
price_house_type = df[df['neighbourhood_group'] ==
                      str(option)].groupby('room_type').mean()['price']
st.dataframe(price_house_type)
