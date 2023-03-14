#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))




from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
from plotly.graph_objs import *
init_notebook_mode()
import numpy as np
import pandas as pd




# importing Orders file
orders = pd.read_csv(r"../input/orders.csv")




y = pd.DataFrame(orders.groupby(["user_id"]).size(),columns=["ordercount"]).reset_index().groupby("ordercount").size().values
x = pd.DataFrame(orders.groupby(["user_id"]).size(),columns=["ordercount"]).reset_index().groupby("ordercount").size().index.values
data = Bar(x = x, y = y,
           marker=dict(color='rgb(221,28,119)',line=dict(color='black',width=0.5,)),opacity=0.6)
layout = Layout(xaxis=dict(title="Total number of orders"),title='# of shoppers for min to max total orders bucket')
fig = Figure(data=[data], layout=layout)
iplot(fig)




#ReARRANGING Orders file data for hourly and daily analysis of total orders
temp = pd.DataFrame({'count':orders.groupby(['order_dow','order_hour_of_day']).size()}).reset_index()
temp.columns = ["Day","Hour","Count"]
temp = temp.pivot(index='Day', columns='Hour', values='Count')
temp




#Heat map for all weekdays for each hour. This will help in understanding the busiest days and hours
import calendar
x = [str(s)+':00' for s in temp.columns.values]
z = [temp.iloc[i].values for i in temp.index.values]
y = list(calendar.day_abbr)[-1:]+list(calendar.day_abbr)[:-1]
data = [Heatmap(z=z,
                x=x,
                y=y,
                colorscale="[[0, 'rgb(253,224,221)', [1, 'rgb(197,27,138)']]")]
layout = Layout(title='Hourly Intensity of Orders on all weekdays')
fig = Figure(data=data, layout=layout)
iplot(fig)




temp = pd.DataFrame({'count':orders.groupby(["order_dow","eval_set"]).size()}).reset_index().pivot(index='order_dow',columns='eval_set',values='count')
temp["current_orders"] = temp["test"]+temp["train"]
temp = temp.iloc[:,[0,3]]
x = list(calendar.day_abbr)[-1:]+list(calendar.day_abbr)[:-1]
data = [Bar(x=x,y=temp['current_orders'],name="Current Orders",
           marker=dict(color='blue',line=dict(color='rgb(8,48,107)',width=1.5,)),opacity=0.6),
        Bar(x=x,y=temp['prior'],name="Prior Orders",
           marker=dict(color='green',line=dict(color='rgb(8,48,107)',width=1.5,)),opacity=0.6)]

annotations1=[dict(x=xi,y=1.5*yi,text=str(yi),showarrow=False) for xi,yi in zip(x,temp['current_orders'])]
annotations2=[dict(x=xi,y=1.1*yi,text=str(yi),showarrow=False) for xi,yi in zip(x,temp['prior'])]
layout = Layout(
    barmode='stack',annotations=annotations1 + annotations2,
    title='Current and Prior orders on all weekdays',
    yaxis=dict(showticklabels=False))
fig = Figure(data=data, layout=layout)
iplot(fig)




order_products_train = pd.read_csv(r"../input/order_products__train.csv")
df = pd.DataFrame(order_products_train.groupby('product_id').size()).reset_index()

temp_df = pd.read_csv(r"../input/order_products__prior.csv",iterator=True,chunksize=100000)
for chunk in temp_df:
    df1 = pd.DataFrame(chunk.groupby('product_id').size()).reset_index()
    df = pd.concat([df,df1],axis=0)
df.columns = ["product_id","#oforders"]
order_count = pd.DataFrame(df.groupby('product_id').agg({'#oforders':np.sum})).reset_index()




aisles = pd.read_csv(r"../input/aisles.csv")
products = pd.read_csv(r"../input/products.csv")
departments = pd.read_csv(r"../input/departments.csv")




products_ordered = order_count.set_index("product_id").join(products.set_index("product_id")).reset_index().set_index("department_id").join(departments.set_index("department_id")).reset_index().set_index("aisle_id").join(aisles.set_index("aisle_id")).reset_index()




temp = pd.DataFrame(products_ordered.groupby(["department_id","department"]).agg({"#oforders":np.sum})).reset_index()
temp.sort_values(by="#oforders", axis=0, ascending=True,inplace=True)
values = list(temp["#oforders"].astype(int))
department_id = list(temp["department_id"])
temp1 = pd.DataFrame(products_ordered.groupby(["department_id","department","aisle"]).agg({"#oforders":np.sum})).reset_index()
temp1.sort_values(by="#oforders", axis=0, ascending=True,inplace=True)




colors = {
35:'rgb(64,0,75)',
34:'rgb(118,42,131)',
33:'rgb(153,112,171)',
32:'rgb(194,165,207)',
31:'rgb(231,212,232)',
30:'rgb(247,247,247)',
29:'rgb(217,240,211)',
28:'rgb(166,219,160)',
27:'rgb(90,174,97)',
26:'rgb(27,120,55)',
25:'rgb(0,68,27)',
24:'rgb(166,206,227)',
23:'rgb(31,120,180)',
22:'rgb(178,223,138)',
21:'rgb(51,160,44)',
20:'rgb(251,154,153)',
19:'rgb(227,26,28)',
18:'rgb(253,191,111)',
17:'rgb(255,127,0)',
16:'rgb(202,178,214)',
15:'rgb(106,61,154)',
14:'rgb(255,255,153)',
13:'rgb(177,89,40)',
12:'rgb(141,211,199)',
11:'rgb(255,255,179)',
10:'rgb(190,186,218)',
9:'rgb(251,128,114)',
8:'rgb(128,177,211)',
7:'rgb(253,180,98)',
6:'rgb(179,222,105)',
5:'rgb(252,205,229)',
4:'rgb(217,217,217)',
3:'rgb(188,128,189)',
2:'rgb(204,235,197)',
1:'rgb(255,237,111)'}




import squarify
x = 0.
y = 0.
width = 100.
height = 100.
rects = squarify.normalize_sizes(values, width, height)
rects = squarify.squarify(rects, x, y, width, height)
shapes = []
shapes0 = []
annotations = []
data = []
counter = 0
for r in rects:
    shapes.append( 
        dict(
            type = 'rect', 
            x0 = r['x'], 
            y0 = r['y'], 
            x1 = r['x']+r['dx'], 
            y1 = r['y']+r['dy'],
            line = dict( width = 2 ),
            fillcolor = colors[department_id[counter]]
        ) 
    )
    x0 = r['x']
    y0 = r['y']
    width0 = r['dx']
    height0 = r['dy']
    val0 = list(temp1[temp1["department_id"]==temp.iloc[counter][0]]["#oforders"])
    aisles0 = list(temp1[temp1["department_id"]==temp.iloc[counter][0]]["aisle"])
    rects0 = squarify.normalize_sizes(val0, width0, height0)
    rects0 = squarify.squarify(rects0, x0, y0, width0, height0)
    count1 = 0
    name = temp.iloc[counter][1]+str((round(val0[count1]/1000,2)))+'K'
    for r0 in rects0:
        shapes0.append( 
            dict(
                type = 'rect', 
                x0 = r0['x'], 
                y0 = r0['y'], 
                x1 = r0['x']+r0['dx'], 
                y1 = r0['y']+r0['dy'],
                line = dict( width = 0.2 ),
                fillcolor = colors[department_id[counter]]
            ))
        trace0 = Scatter(
            x = [ r0['x']+(r0['dx']/2) for r0 in rects0 ], 
            y = [ r0['y']+(r0['dy']/2) for r0 in rects0 ],
            name = name,
            text = aisles0,
            hoverinfo="text+name") 
        data = data + [trace0]
        count1+=1
    annotations.append(
        dict(
            x = r['x']+(r['dx']/2),
            y = r['y']+(r['dy']/2),
            text = temp.iloc[counter][1],
            font=dict(family='Open Sans',color='black',size=counter+1),
            bgcolor='white' ,
            showarrow = False
        )
    )        
    counter = counter + 1
    if counter >= len(colors):
        counter = 0
layout = dict(
    height=800, 
    width=800,
    xaxis=dict(visible=False,showgrid=False,zeroline=False),
    yaxis=dict(visible=False,showgrid=False,zeroline=False),
    shapes=shapes+shapes0,
    annotations=annotations,
    showlegend = False,
    hovermode='closest',
    title = "Departments, aisles and total number of products"
)

figure = dict(data=data,layout=layout)
iplot(figure, filename='squarify-treemap')




order_products_train = pd.read_csv(r"../input/order_products__train.csv")
df = pd.DataFrame(order_products_train.groupby('order_id').agg({'reordered':[np.sum,len]})).reset_index()
temp_df = pd.read_csv(r"../input/order_products__prior.csv",iterator=True,chunksize=100000)
for chunk in temp_df:
    df1 = pd.DataFrame(chunk.groupby('order_id').agg({'reordered':[np.sum,len]})).reset_index()
    df = pd.concat([df,df1],axis=0)
df.columns = ["order_id","reordered_items","cart_size"]
df["%reordered"] = round(df["reordered_items"]*100/df["cart_size"],2)




trace0 = Histogram(x=df["%reordered"],
                  autobinx=False,
                  xbins=dict(start=0,end=100,size=10),
                  marker=dict(color = "rgb(221,28,119)",line=dict(width=1)),error_y=dict(visible=True),opacity=0.5,
                  cumulative=dict(enabled=True,direction="decreasing"))
layout = Layout(width=700,height=700,xaxis=dict(title="% reordered items in terms of order size"),
                yaxis=dict(title="No. of Orders"),
                title = "Decreasing cumulative orders with % reordered items against order size")
fig = Figure(data=[trace0],layout=layout)
iplot(fig)




User_Id_Most_orders = orders.groupby("user_id").size().nlargest(10).index.values[0]
most_active_shopper = pd.DataFrame(orders[orders["user_id"]==User_Id_Most_orders].groupby(["order_dow","order_hour_of_day"]).size()).reset_index()
most_active_shopper = most_active_shopper.pivot(index='order_dow', columns='order_hour_of_day', values=0).fillna(0)

import calendar
x = [str(s)+':00' for s in most_active_shopper.columns.values]
z = [most_active_shopper.iloc[i].values for i in most_active_shopper.index.values]
y = list(calendar.day_abbr)[-1:]+list(calendar.day_abbr)[:-1]
data = [Heatmap(z=z,
                x=x,
                y=y,colorscale="[[0, 'rgb(253,224,221)', [1, 'rgb(197,27,138)']]")]
layout = Layout(title='Hourly Intensity of Orders on all weekdays')
fig = Figure(data=data, layout=layout)
iplot(fig)

