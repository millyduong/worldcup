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
matches=load_data('matches.csv')
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
df=load_data('sunburst_df.csv')

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
df=load_data('mean_attendance.csv')
#st.dataframe(df)
st.subheader('Average Number of Attendees Per Match Over the Years')
teams = st.multiselect("Show teams you'd like to compare?", df['Teams'].unique())
new_df = df[df['Teams'].isin(teams)]

if len(teams) > 0 :
    fig=px.line(new_df, x='Year', y="Attendance", color='Teams')

    fig.add_vrect(x0=1938, x1=1950, line_width=0, fillcolor="black", opacity=0.3)
    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=[1930, 1934, 1938, 1950, 1954, 1958, 1962, 1966, 1970, 1974, 1978,
        1982, 1986, 1990, 1994, 1998, 2002, 2006, 2010, 2014]
        )
    )
    st.plotly_chart(fig)

if st.checkbox('Show raw data', key='raw3'):
    st.subheader('Raw data')
    st.write(df)


#Correlation between history performance and attendance
df=load_data('correlation.csv')
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

##############################################
#       Sponsorship over the years           #
##############################################

## Library imports

import pandas_datareader.data as web
from plotly.subplots import make_subplots
import plotly.graph_objects as go

## List of sponsors

sponsors = {
    "1978": ["Coca-Cola", "Canon", "Philips", "Gillette", "KLM", "Café de Brasil"], 
    "1982": ["Canon", "Coca-Cola", "Fujifilm", "Gillette", "Iveco", "JVC", "Metaxa", "RJ Reynolds", "Seiko"], 
    "1986": ["Anheuser-Busch", "Bata", "Canon", "Cinzano", "Coca-Cola", "Fujifilm", "Gillette", "JVC", "Opel", "Philips", "RJ Reynolds", "Seiko"], 
    "1990": ["Coca-Cola", "Alfa Romeo", "Anheuser-Busch", "Canon", "Fujifilm", "Gillette", "JVC", "Philips", "Snickers", "Vini D’Italia"], 
    "1994": ["Canon", "Coca-Cola", "Energizer", "Fujifilm", "General" "Motors", "Gillette", "JVC", "Mastercard", "McDonald's", "Philips", "Snickers"], 
    "1998": ["Adidas", "Coca-Cola", "Anheuser-Busch", "Canon", "Fujifilm", "Gillette", "JVC", "Mastercard", "McDonald's", "Opel", "Philips", "Snickers"], 
    "2002": ["Adidas", "Coca-Cola", "Hyundai", "Sony", "Anheuser-Busch", "Avaya", "Fujifilm", "Gillette", "JVC", "Korea Telekom/NTT", "Mastercard", "McDonald's", "Philips", "Toshiba", "Yahoo"], 
    "2006": ["Adidas", "Coca-Cola", "Emirates", "Hyundai", "Sony", "Anheuser-Busch", "Avaya", "Continental", "Deutsche Telekom", "Fujifilm", "Gillette", "Mastercard", "McDonald's", "Philips", "Toshiba"], 
    "2010": ["Yahoo", "Anheuser-Busch", "Castrol", "Continental", "McDonald's", "MTN", "Satyam"], 
    "2014": ["Anheuser-Busch", "Castrol", "Continental", "Johnson & Johnson", "McDonald's", "Oi", "Seara", "Yingli Solar"],
    "2018": ["Anheuser-Busch", "Hisense", "Sony", "McDonald's", "Mengniu", "Yadea", "Rostelecom", "Diking", "LUCI"]
}

## Unique sponsors
s = []
for y in sponsors:
  for n in sponsors[y]:
    if n not in s:
      s.append(n)

## Dataframe with 0 as default value
sp = pd.DataFrame(index=s, columns = sponsors.keys(), data=0)

## Fill dataframe with sponsorship information
for y in sponsors:
  for n in sponsors[y]:
    sp[y][n] = 1

sp.reset_index(inplace=True)
sp.rename(columns={"index":"sponsor"}, inplace=True)

## Stock trading symbols for publicly traded companies that sponsored the campionship
symbols = ['KO', 'CAJ', 'PHG', 'N/A', 'KLMR', 'N/A', 'FUJIY', 'N/A', 'N/A', 'N/A', 'N/A', 'SEKEY', 'BUD', 'N/A', 'N/A', 'N/A', 'N/A', 'HSY', 'N/A', 'ENR', 'GM', 'MA', 'MCD', 'ADDYY', 'HYMTF', 'SNE', 'AVYA', 'KT', 'TOSBF',
           'N/A', 'N/A', 'CLR', 'N/A', 'DTEGY', 'N/A', 'N/A', 'SAY', 'JNJ', 'N/A', 'N/A', 'HISEF', 'N/A','N/A','N/A','N/A','N/A']
sp['symbol'] = symbols

## Cull the list
sp = sp[sp['symbol']!='N/A']

## Create dictionary of dataframes with stock information from all the sponsors

@st.cache
def get_sponsor_stock_data(sp_list):
    s_dict = {}
    for k in sp_list['sponsor']:
        s_dict[k] = web.DataReader(sp_list[sp_list['sponsor']==k]['symbol'], start='1978', end='2020', data_source='yahoo')
        s_dict[k].columns = s_dict[k].columns.droplevel('Symbols')
        s_dict[k].drop(columns = ['Adj Close', 'High', 'Low', 'Open'], inplace=True)
        s_dict[k] = s_dict[k].resample('M').mean()
    return s_dict

stock = get_sponsor_stock_data(sp)

## Graph the info

st.subheader('Stock price (mothly average)')

@st.cache(allow_output_mutation=True)
def draw_from_stock_dict(stock_dict):

    fig = make_subplots(
        rows=12, cols=2,
        subplot_titles=[k for k, v in stock_dict.items()]
    )

    for i, k in enumerate(stock_dict.keys()):
        r = (i+2)//2
        c = ((i+2)%2)+1
        
        fig.add_trace(go.Scatter(x=stock_dict[k].index, y=stock_dict[k]["Close"], mode='lines'),
                row=r, col=c)
        t = sp.set_index('sponsor').T.reset_index()
        years = pd.to_datetime(t[t[k] == 1]['index'].values) + pd.DateOffset(180)
        
        for y in years:
            fig.add_vline(x=y, line_width=0.5, line_color="orangered", row=r, col=c)
        fig.update_xaxes(
            range = [years[0] - pd.DateOffset(365), years[-1] + pd.DateOffset(365)],
            tickvals = years,
            ticktext = years.year,
            title = '',row=r, col=c
        )

        fig.update_yaxes(
            title = 'Share Price (USD)',row=r, col=c
        )
                        
    fig.update_layout(height=3200, width=1000, title_text="Share price of publicly traded sponsors", showlegend=False)
    return(fig)

st.plotly_chart(draw_from_stock_dict(stock))

if st.checkbox('Show raw data', key='raw5'):
    st.subheader('Raw data')
    st.write(sp)
    for k in stock.keys():
        st.write(stock[k])


##############################################
#        International performance           #
##############################################

## Read the file
inter = pd.read_csv('international_matches.csv')

## Clean data
inter['year'] = pd.DatetimeIndex(inter['date']).year

g1 = inter.groupby(['home_team','year']).agg({'home_score':['sum', 'count'], 'away_score':'sum'})
g1.columns=['scored', 'played', 'against']
g1.index.set_names(['team', 'year'], inplace=True)

g2 = inter.groupby(['away_team','year']).agg({'home_score':'sum', 'away_score':['sum', 'count']})
g2.columns=['against', 'scored', 'played']
g2.index.set_names(['team', 'year'], inplace=True)

gby = g1.add(g2, fill_value=0).reset_index()


## calculate averages
gby['r_against'] = gby['against'].rolling(8).mean()
gby['r_scored'] = gby['scored'].rolling(8).mean()
gby['r_played'] = gby['played'].rolling(8).mean()


st.subheader('Goal production throughout the years')

## Selection boxes
team = st.selectbox("Choose a team", gby['team'].unique(), index=gby['team'].unique().tolist().index('Vietnam'))
played = st.checkbox("Include matches count?")

th = gby[gby['team'] == team]

## Graph
figure = go.Figure()

figure.add_trace(go.Scatter(x=th['year'], y=th['r_scored'], mode='lines', name='Goals scored'))
figure.add_trace(go.Scatter(x=th['year'], y=th['r_against'], mode='lines', name='Goals against'))

if played:
    figure.add_trace(go.Scatter(x=th['year'], y=th['played'], mode='lines', name='Matches played'))

st.plotly_chart(figure)

if st.checkbox('Show raw data', key='raw6'):
    st.subheader('Raw data')
    st.write(th)