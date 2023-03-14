#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.plotly as py
import plotly.offline as py
import squarify
import seaborn as sns
import geojson
import IPython.display

py.init_notebook_mode(connected=True)


# In[2]:


items = pd.read_csv('../input/favorita-grocery-sales-forecasting/items.csv')
holiday = pd.read_csv("../input/favorita-grocery-sales-forecasting/holidays_events.csv")
stores = pd.read_csv("../input/favorita-grocery-sales-forecasting/stores.csv")
oil = pd.read_csv("../input/favorita-grocery-sales-forecasting/oil.csv")
transaction = pd.read_csv("../input/favorita-grocery-sales-forecasting/transactions.csv", parse_dates=['date'])
train = pd.read_csv("../input/favorita-grocery-sales-forecasting/train.csv", nrows=6000000, parse_dates=['date'])


# In[3]:


# Take a look at the data. 
# 22 cities, 16 states
stores.head(5)


# In[4]:


stores['city'] = list(map(str.upper, stores['city']))
stores['state'] = list(map(str.upper, stores['state']))


# In[5]:


number_of_stores_per_city = {}
store_per_type = stores.groupby(['city','type']).store_nbr.size().unstack().fillna(0)
store_per_cluster = stores.groupby(['city','cluster']).store_nbr.size().unstack().fillna(0)

for i, c in enumerate(stores.city.value_counts().index):
    number_of_stores_per_city[c] = [stores.city.value_counts().values[i]]
    number_of_stores_per_city[c].extend(store_per_type.loc[c].values)
    number_of_stores_per_city[c].extend(store_per_cluster.loc[c].values)

cities = [s.upper() for s in stores.city.unique()]
states = [s.upper() for s in stores.state.unique()]


# In[6]:



with open("../input/ecuador-geo-info/ecuador.geojson") as json_file:
    json_data = geojson.load(json_file)


# In[7]:


patches, lons, lats, text = [], [], [], []
for k, feature in enumerate(json_data['features']):
    state = feature['properties']['DPA_DESPRO']
    city = feature['properties']['DPA_DESCAN']
    
    if(city not in cities):
        continue
    
    m, M = np.array(feature["geometry"]["coordinates"][0][0])[:,0].max(), np.array(feature["geometry"]["coordinates"][0][0])[:,0].min()
    lons.append(0.5*(m+M))
    
    m, M = np.array(feature["geometry"]["coordinates"][0][0])[:,1].max(),np.array(feature["geometry"]["coordinates"][0][0])[:,1].min()
    lats.append(0.5*(m+M)) 
    
    num_of_stores = number_of_stores_per_city[city]
    t = "State: " + state + '<br>' + "City: " + city + '<br> ' +         "Number of stores: " + str(num_of_stores[0])
    
    text.append(t)
    
    sub = {'type':"FeatureCollection"}
    sub['features'] = [json_data['features'][k]]
    sub['number'] = num_of_stores
    patches.append(sub) # patches is a list of dictionary. 


# In[8]:


# A list of layers, each layer cooresponds to a city. 
# Representing how many stores opened in each city. 
stores_per_city_layers = []
mapbox_access_token = "pk.eyJ1IjoiYmF0byIsImEiOiJjamJwZzRvaGE2MTljMzJtcjhzaDJvaXFxIn0.TkTLg13Af-ERPjOWzB-BFQ"

for i in range(len(patches)):
    num_stores_each_city = dict(
        sourcetype = "geojson",
        source = patches[i],
        type = "fill",
        opacity= (1-0.3)/(18-1) * (patches[i]['number'][0]-18)+1,
        color = 'rgb(163,22,190)',
    )
    stores_per_city_layers.append(num_stores_each_city)

# per_city_layout: the layout for the button representing how 
# many stores in each city. 
per_city_layout = dict(
        layers=stores_per_city_layers,
        accesstoken=mapbox_access_token,
        bearing=0,
        center=dict(
            lat=-2,
            lon=-78
        ),
        pitch=0,
        zoom=5.2,
        style='light'
    )


# In[9]:


# Create a list of dict, each dict cooresponds to a relayout button for a type
store_type_buttons= []
types_of_stores = ['A','B','C','D','E']


for i in range(1, len(types_of_stores)+1):
    type_per_city_layers = []
    for j in range(len(patches)):
        type_i_for_city_j_layer=dict(
            sourcetype = "geojson",
            source = patches[j],
            type = "fill",
            opacity= patches[j]['number'][i]/6 * 3,
            color = 'rgb(163,22,190)'
        )
        
        type_per_city_layers.append(type_i_for_city_j_layer)
    
    per_type_layout = dict(
        layers=type_per_city_layers,
        accesstoken=mapbox_access_token,
        bearing=0,
        center=dict(
            lat=-2,
            lon=-78
        ),
        pitch=0,
        zoom=5.2,
        style='light'
    )
    
    store_type_buttons.append(
        dict(
            label="Type "+types_of_stores[i-1],
            method="relayout",
            args = ['mapbox', per_type_layout],
        )
        
        
    )


# In[10]:


stores_cluster_buttons = []
cluster_of_stores = range(6,17+6)

for i in cluster_of_stores:
    cluster_per_city_layers=[]
    for j in range(len(patches)):
        cluster_i_for_city_j_layer=dict(
            sourcetype = "geojson",
            source = patches[j],
            type = "fill",
            opacity= patches[j]['number'][i]/4 * 3,
            color = 'rgb(163,22,190)'
        )
        
        cluster_per_city_layers.append(cluster_i_for_city_j_layer)
        
    per_cluster_layout = dict(
        layers=cluster_per_city_layers,
        accesstoken=mapbox_access_token,
        bearing=0,
        center=dict(
            lat=-2,
            lon=-78
        ),
        pitch=0,
        zoom=5.2,
        style='light'
    )
    
    stores_cluster_buttons.append(
        dict(
            label="Cluster "+str(i-6),
            method="relayout",
            args = ['mapbox', per_cluster_layout],
        )
        
        
    )


# In[11]:


stores_trace = go.Data([
    go.Scattermapbox(
        visible=True,
        lat=lats,
        lon=lons,
        text = text,
        mode='markers',
        name= "Number of stores per city",
        hoverinfo='text',
        showlegend=False,
        marker=dict(size=5, opacity=0)
    )
])

updatemenu = list([
    
    dict(type="buttons",
         active=0,
        buttons=list([
            dict(label="Number of stores",
                method='relayout',
                args= ['mapbox', per_city_layout],
                ),
            
        ]),
        x = -0.01, y=0.9
    ),
    
    dict(type="dropdown",
        buttons= store_type_buttons,
        x = -0.01, y=0.8
    ),
    
    dict(type="dropdown",
        buttons=stores_cluster_buttons,
        x=-0.01, y=0.7)
    
])

stores_layout = go.Layout(
    title="Number of Stores per city",
    height=600,
    autosize=True,
    mapbox=dict(
        #layers=stores_per_city,
        accesstoken=mapbox_access_token,
        bearing=0,
        center=dict(
            lat=-2,
            lon=-78
        ),
        pitch=0,
        zoom=5.2,
        style='light'
    ),
    updatemenus=updatemenu
)


stores_fig = dict(data=stores_trace, layout=stores_layout)
py.iplot(stores_fig)


# In[12]:


transaction.head(5)


# In[13]:


transaction['year'] = pd.DatetimeIndex(transaction['date']).year.astype(np.uint16)
transaction['month'] = pd.DatetimeIndex(transaction['date']).month.astype(np.uint8)
transaction['day'] = pd.DatetimeIndex(transaction['date']).day.astype(np.uint8)


# In[14]:


which_store_trace = []
number_of_stores = 54
transaction_per_month_trace=[]
per_month_transaction = transaction.groupby(['store_nbr', 'year', 'month']).mean()['transactions']

buttons = [
    
    dict(
        label = "None",
        method="update",
        args=[{'visible':[False]*number_of_stores*2}]
    ),
    dict(
        label = "all stores Daily",
        method='update',
        args=[{'visible':[True]*number_of_stores+[False]*number_of_stores},
             {"title": "All stores daily transaction from 2013 to 2017"}]
    ),
]

monthly_buttons = [
    dict(
        label = "all stores Monthly",
        method='update',
        args=[{'visible':[False]*number_of_stores+[True]*number_of_stores},
             {"title": "All stores monthly transaction from 2013 to 2017"}]
    ),
    dict(
        label = "None",
        method="update",
        args=[{'visible':[False]*number_of_stores*2}]
    )
    
]
for i in range(1, number_of_stores+1):
    visible = [False]*number_of_stores*2
    
    which_store_trace.append(go.Scatter(
        x = transaction.loc[lambda df:df.store_nbr==i].date.dt.strftime("%Y-%m-%d"),
        y = transaction.loc[lambda df:df.store_nbr==i].transactions.values,
        name = "store "+str(i)+" daily transactions from 2013 to 2017",
        visible=False,
    ))
    
    transaction_per_month_trace.append(go.Scatter(
        x = transaction.loc[lambda df:(df.store_nbr==i)].loc[lambda df:(df.day==24)].date.dt.strftime("%Y-%m-%d"),
        y = per_month_transaction.iloc[per_month_transaction.index.get_level_values('store_nbr')==i].values,
        name = "store "+str(i)+" monthly transactions from 2013 to 2017"
    ))
    
    visible[i-1]=True
    buttons.append(dict(
        label = "store " + str(i) + " Daily",
        method="update",
        args=[{'visible':visible},
             {'title': "Store "+str(i)+" transactions from 2013 to 2017"}]
    ))
    visible = [False]*number_of_stores*2
    
    visible[number_of_stores+i-1]=True
    monthly_buttons.append(dict(
        label = "store " + str(i) + " Monthly",
        method = 'update',
        args = [{'visible':visible},
               {'title':"Store "+str(i)+" monthly transaction from 2013 to 2017"}]
    ))

updatemenu = list([
    dict(
        type="dropdown",
        buttons = buttons,
        x = 1.2, y = 0.7
    ),
    
    dict(
        type="dropdown",
        buttons = monthly_buttons,
        x = 1.2,
        y = 0.9,
    )
                  
                  
])


transaction_per_store_layout = dict(title="Transaction per store", 
                                   showlegend=False, 
                                   updatemenus=updatemenu,
                                   yaxis=dict(title="Number of transaction"),
                                    xaxis=dict(rangeslider=dict(), type='date')
                                   )

transaction_per_store_fig = dict(data=which_store_trace+transaction_per_month_trace, layout=transaction_per_store_layout)
py.iplot(transaction_per_store_fig)


# In[15]:


oil.head(5)


# In[16]:


trace = go.Scatter(
    name= "Oil prices",
    x=oil['date'], 
    y=oil['dcoilwtico'], 
    mode='lines+markers', 
    marker=dict(size=2, color = 'rgba(0, 152, 0, .8)'),
    fill='tonexty')
data = [trace]
layout = dict(title="Crude Oil prices from 2013.1 to 2017.8 at Oklahoma", 
             yaxis=dict(title="Daily Oil price"))
fig = go.Figure(data = data, layout = layout)


# In[17]:


py.iplot(fig)


# In[18]:


items.head(10)


# In[19]:


trace_item = go.Bar(
    y = items.family.value_counts(ascending=True).index,
    x = items.family.value_counts(ascending=True).values,
    marker=dict(
        color=items.family.value_counts(ascending=True).values,
        colorscale="Rainbow"
    ),
    orientation='h'
)

layout_item = dict(
    title="Counts of items per family",
    width=800, height=800,
    margin=dict(l=140)
)

fig_item = go.Figure(data=[trace_item], layout=layout_item)

py.iplot(fig_item)


# In[20]:


trace_item_perishable = go.Bar(
    y = items.loc[lambda df:df.perishable==1].family.value_counts(ascending=True).index,
    x = items.loc[lambda df:df.perishable==1].family.value_counts(ascending=True).values,
    orientation='h',
    name = 'perishable',
    marker=dict(color="#F39C12")
)


trace_item_unperishable=go.Bar(
    y = items.loc[lambda df:df.perishable==0].family.value_counts(ascending=True).index,
    x = items.loc[lambda df:df.perishable==0].family.value_counts(ascending=True).values,
    orientation='h',
    name = "non-perishable",
    marker=dict(color="#3498DB")
)

layout_perish = dict(
    width=800, height=800,
    title = "Perishable item per family vs non-Perish item per family",
    margin=dict(l=140)
)

fig = go.Figure(data=[trace_item_perishable, trace_item_unperishable], layout=layout_perish)

py.iplot(fig)


# In[21]:


pie_perish_trace = go.Pie(
    labels = ['non-perishable', 'perishable'], 
    values = items.perishable.value_counts().values,
    marker=dict(colors=["#3498DB", "#F39C12"]),
    text=['non-perishable','perishable'],
    hoverinfo="label+value",
    domain = dict(x=[0, 0.48], y=[0.5, 1]),
    showlegend = False
)

pie_class_trace = go.Pie(
    labels = items['class'].value_counts().index[0:5],
    values = items['class'].value_counts().values[0:5],
    domain = dict(x=[0.52, 1], y=[0.5,1]),
    showlegend = False,
    text = ["class "+str(x) for x in items['class'].value_counts().index[0:5]],
    hoverinfo="text+percent",
    textinfo="value+text"
)

item_pie_layout = dict(
    width=800, height=700,
    title="Percentage of perishable items          Most frequent classes of the items"
)

pie_perish_fig = go.Figure(data=[pie_perish_trace, pie_class_trace], layout=item_pie_layout)
py.iplot(pie_perish_fig)


# In[22]:


item_counts_per_family = items['family'].value_counts().values
item_index_per_family = items['family'].value_counts().index

x = 0
y = 0
width =250
height=250

normed_family = squarify.normalize_sizes(item_counts_per_family, width, height)
rects_family = squarify.squarify(normed_family, x, y, width, height)


# In[23]:


shape_family = []
annotation_family = []
counter = 0
color_brewer = sns.diverging_palette(220, 20, n=33)
color_template = ["rgba" + str(tuple(x)) for x in color_brewer]

for r in rects_family:
    shape_family.append(
        dict(
            type='rect',
            x0 = r['x'],
            y0 = r['y'],
            x1 = r['x']+r['dx'],
            y1 = r['y']+r['dy'],
            line = dict(width=2),
            fillcolor = color_template[counter]
        )
    )
    
    annotation_family.append(
        dict(
            x = r['x']+(r['dx']/2),
            y = r['y']+(r['dy']/1.5),
            text = item_index_per_family[counter],
            showarrow=False,
            align="center",
            font=dict(
                color="blue",
                size=max(1, r['dx']/4)
            ),
        )
    )
    
    counter+=1


# In[24]:


treemap_family_item_info = go.Scatter(
    x = [r['x']+(r['dx']/2) for r in rects_family],
    y = [r['y']+(r['dy']/2) for r in rects_family],
    text = [str(v) for v in item_counts_per_family],
    mode = 'text',
)

treemap_all_item_layout = dict(
    height=900, width=900,
    shapes = shape_family,
    hovermode='closest',
    annotations = annotation_family,
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=False),
)


treemap_item_fig = dict(data=[treemap_family_item_info],
                       layout=treemap_all_item_layout)

py.iplot(treemap_item_fig)


# In[25]:


print(holiday.shape)
holiday.iloc[0:30]


# In[26]:


holiday_type_trace = go.Pie(
    labels = holiday.type.value_counts().index,
    values = holiday.type.value_counts().values,
    domain = dict(x=[0,0.49], y=[0, 0.49]),
    showlegend = False,
    text = holiday.type.value_counts().index,
    hoverinfo = "text+percent",
    textinfo = "value+text",
    hole = 0.3
)

holiday_locale_trace=go.Pie(
    labels = holiday.locale.value_counts().index,
    values = holiday.locale.value_counts().values,
    domain = dict(x=[0.52,1], y=[0, 0.49]),
    showlegend = False,
    text = holiday.locale.value_counts().index,
    hoverinfo="text+percent",
    textinfo="value+text",
    hole = 0.3
)

holiday_locale_name_trace=go.Pie(
    labels = holiday.locale_name.value_counts().index,
    values = holiday.locale_name.value_counts().values,
    domain = dict(x=[0,0.48], y=[0.51, 1]),
    showlegend = False,
    text = holiday.locale_name.value_counts().index,
    hoverinfo="text+percent",
    textinfo="value",
    hole = 0.3
)

holiday_transferred_trace=go.Pie(
    labels = holiday.transferred.value_counts().index,
    values = holiday.transferred.value_counts().values,
    domain = dict(x=[0.52, 1], y=[0.51, 1]),
    showlegend = False,
    text = holiday.transferred.value_counts().index,
    hoverinfo="text+percent",
    textinfo="text+value",
    hole = 0.3
)

holiday_layout = dict(
    width=800, height=800,
    annotations= [
        dict(
            font=dict(
                size=15,
                family="Droid Sans"
            ),
            text = "locale name",
            x = 0.18, y=0.78,
            showarrow=False
        ),
        
        dict(
            font=dict(
                size=15,
                family="Droid Sans"
            ),
            text = "Transferred",
            x = 0.82, y=0.78,
            showarrow=False
        ),
        
        dict(
            font=dict(
                size=15,
                family="Droid Sans"
            ),
            text = "Type",
            x = 0.215, y=0.23,
            showarrow=False
        ),
        
        dict(
            font=dict(
                size=15,
                family="Droid Sans"
            ),
            text = "Locale",
            x = 0.80, y=0.23,
            showarrow=False
        ),
    ]
)


holiday_type_fig = go.Figure(data=[holiday_type_trace, holiday_locale_trace,
                                  holiday_locale_name_trace, holiday_transferred_trace], 
                            layout = holiday_layout)

py.iplot(holiday_type_fig)


# In[27]:




