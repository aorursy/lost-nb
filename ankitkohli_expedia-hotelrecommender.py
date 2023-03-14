# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

customerData = pd.read_csv("../input/train.csv",chunksize=1000000)


aggregation = {
    'is_booking':{
        'bookings':'sum'
    },
    'is_booking':{
        'clicks':'count'
    }
}



aggregations=[]
for chunk in customerData:
    agg = chunk.groupby(["srch_destination_id","hotel_cluster"]).agg(aggregation)
    agg.reset_index(inplace=True)
    aggregations.append(agg)
    print('.',end='')
print('')
aggregations = pd.concat(aggregations, axis=0)
aggregations.head(10
