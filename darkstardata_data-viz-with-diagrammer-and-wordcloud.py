#!/usr/bin/env python
# coding: utf-8



# Enables R line magic (%R) and cell magic (%%R)
get_ipython().run_line_magic('load_ext', 'rpy2.ipython')

# r enables calls to r objects and pandas2ri allows conversion both ways
from rpy2.robjects import r, pandas2ri    

# Ignore rpy2 RRuntimeWarning complaining about packages, they work fine!
import warnings
from rpy2.rinterface import RRuntimeWarning
warnings.filterwarnings("ignore", category=RRuntimeWarning)

# Enables rendering a string of HTML code as an image
from IPython.display import display, HTML




get_ipython().run_cell_magic('R', '', '\nlibrary(DiagrammeR)\nlibrary(DiagrammeRsvg) # provides export_svg command and thats it\n\nplot.script <- paste("\n    digraph \'ER-D\' {\n    \n        graph [layout = neato]\n        \n        node [style = filled, penwidth = 3]\n        \n        node [color = royalblue, fillcolor = cornflowerblue, shape = egg, width=01.00]\n        aisles departments orders products order_products__\n        \n        node [color = forestgreen, fillcolor = darkseagreen, shape = box, width=0.40, height=0.30, fontsize=9]\n        \n        edge [color = grey, arrowhead=none]\n        aisles -> {aisle_id aisle}\n        departments -> {department_id department}\n        orders -> {order_id user_id eval_set order_number order_dow order_hour_of_day days_since_prior_order}\n        products -> {product_id product_name aisle_id department_id}\n        order_products__ -> {order_id product_id add_to_cart_order reordered}\n    }\n")\nsvg <- export_svg(grViz(plot.script))')




display(HTML(r.svg[0]))




get_ipython().run_cell_magic('R', '', '\nlibrary(data.table) # for fread()\nlibrary(dplyr) # for lots data table manipulation of goodies (joins, group_by, etc)\n\norders <- fread(\'/Users/andrew/Desktop/data/orders.csv\')\nproducts <- fread(\'/Users/andrew/Desktop/data/products.csv\')\naisles <- fread(\'/Users/andrew/Desktop/data/aisles.csv\')\ndepartments <- fread(\'/Users/andrew/Desktop/data/departments.csv\')\norder_products <- fread(\'/Users/andrew/Desktop/data/order_products__train.csv\')\n\n\n# Combine aisle and deparment data with product data\ntmp <- products %>% \n    group_by(department_id, aisle_id) %>% \n    summarise(n_items=n()) %>% \n    left_join(departments,by="department_id") %>% \n    left_join(aisles,by="aisle_id")\n\n# Combine with order_products__(train) to find total items sold in each aisle as n_orders\ngoods<-order_products %>% \n  group_by(product_id) %>% \n  summarize(count=n()) %>% \n  left_join(products,by="product_id") %>% \n  ungroup() %>% \n  group_by(department_id,aisle_id) %>% \n  summarize(n_orders = sum(count)) %>% \n  left_join(tmp, by = c("department_id", "aisle_id")) %>% \n  mutate(onesize = 1)')




# 
get_ipython().run_line_magic('R', 'goods %>% head(1)')




get_ipython().run_line_magic('R', 'goods_tmp <- goods[ ,c(6,3)] # select n_orders and aisles column from goods in R')
get_ipython().run_line_magic('R', 'goods_tmp %>% head(5)')




from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")

path = '/Users/andrew/Desktop/data/'
im_path = path + 'instacart_color.png'
im_out_path = 'instacart_wc.png'


get_ipython().run_line_magic('R', 'goods_tmp <- goods[ ,c(6,3)] # select n_orders and aisles column from goods in R')

goods = pandas2ri.ri2py(r.goods_tmp) # make goods_tmp into pandas data frame in Python

# convert data-frame to list of lists for consumption
#  by wordcloud.generate_from_frequencies(frequencies=d)
d = goods.values.tolist()

# set stop words
stopwords = set(STOPWORDS)
stopwords.add("said")

# read the mask image
instacart_image = np.array(Image.open(im_path))

# create coloring from image
image_colors = ImageColorGenerator(instacart_image)

# create wordcloud object
wc = WordCloud(background_color="white", max_words=134, mask=instacart_image,
               stopwords=stopwords)

# get word counts directly from list of lists
wc.generate_from_frequencies(frequencies=d)



# plot figure
plt.figure()
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")

# plot figure
plt.figure()
plt.imshow(wc.recolor(color_func=image_colors), interpolation="bilinear")
plt.axis("off")

# store to file
wc.to_file(im_out_path)

# plot figure
plt.figure()
plt.imshow(instacart_image, cmap=plt.cm.gray, interpolation='bilinear')
plt.axis("off")

# show plot
plt.show()

