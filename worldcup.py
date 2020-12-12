import numpy as np
import plotly.express as px
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt 

# Write text header
st.header('WORLD CUP ANALYSIS')

#LOAD DATA
@st.cache
def load_data(path):
    data = pd.read_csv(path, encoding='ISO 8859-15')
    return data

data_load_state = st.text('Loading data...')
matches=load_data('/Users/millyduong/Desktop/demo4/matches.csv')
data_load_state.text("Done!")


#st.dataframe(matches)

#SQL TO QUERY RELEVANT DATA
c = matches['Year'].unique()

my_list = []
for i in c:
  a = matches[matches['Year']==i][['Home Team Name','Home Team Goals']].groupby(['Home Team Name'], as_index=False).sum()
  b = matches[matches['Year']==i][['Away Team Name','Away Team Goals']].groupby(['Away Team Name'], as_index=False).sum()

  a.rename(columns={'Home Team Name':'Away Team Name', 'Home Team Goals' : 'Away Team Goals'}, inplace=True)
  result = pd.concat([a,b])
  result = result.groupby(['Away Team Name'],  as_index=False).sum()
  d = {'Country': result['Away Team Name'], 'Year': i, 'Goals' : result['Away Team Goals']}
  df2 = pd.DataFrame(data=d)
  my_list.append(df2)
df3 = pd.concat(my_list)

#DRAW CHOROPLETH MAP 
fig = px.choropleth(
    df3[::1], 
    locations= 'Country', 
    locationmode= 'country names', 
    color= 'Goals',
    hover_name= 'Country',
    hover_data= ['Goals'], 
    animation_frame= 'Year',
    color_continuous_scale=px.colors.sequential.Teal
)

fig.update_layout(
    #title_text =   "Countries participated in World Cup up to 2014",
    title_x = 0.5,
    geo= dict(
        showframe= False,
        showcoastlines= False,
        projection_type = 'equirectangular'
    )
)
st.subheader('Countries participated in World Cup up to 2014')
st.plotly_chart(fig)

if st.checkbox('Show raw data', key='raw1'):
    st.subheader('Raw data')
    st.write(matches)

#SUNBURST CHART
#LOAD DATA
df=load_data('/Users/millyduong/Desktop/demo4/sunburst_df.csv')

df = pd.DataFrame(
    dict(Goals=df['Goals'], Year=df['Year'], Country=df['Country'])
)

#st.dataframe(df)
fig = px.sunburst(df, path=['Country', 'Year'], maxdepth= 2, values = df['Goals'])
st.subheader('Top 7 Countries by wins')
st.plotly_chart(fig)

if st.checkbox('Show raw data',key='raw2'):
    st.subheader('Raw data')
    st.write(df)


#AVERAGE NUMBER OF ATTENDEES PER MATCH OVER THE YEARS
#LOAD DATA
df=load_data('/Users/millyduong/Desktop/demo4/mean_attendance.csv')
#st.dataframe(df)

fig=px.line(df, x='Year', y="Attendance", color='Teams')
fig.update_layout(
    xaxis=dict(
        tickmode='array',
        tickvals=[1930, 1934, 1938, 1950, 1954, 1958, 1962, 1966, 1970, 1974, 1978,
       1982, 1986, 1990, 1994, 1998, 2002, 2006, 2010, 2014]
    )
)
st.subheader('Average Number of Attendees Per Match Over the Years')
st.plotly_chart(fig)

if st.checkbox('Show raw data', key='raw3'):
    st.subheader('Raw data')
    st.write(df)


#Correlation between history performance and attendance
df=load_data('/Users/millyduong/Desktop/demo4/correlation.csv')
#st.dataframe(df)
fig, ax = plt.subplots()
x = df['TotalGoals']
y = df['Attendance']
ax = plt.scatter(x, y)
plt.xlabel("Total goals")
plt.ylabel("Attendance")
st.subheader('Correlation between performance and attendance')
st.pyplot(fig)

if st.checkbox('Show raw data', key='raw4'):
    st.subheader('Raw data')
    st.write(df)