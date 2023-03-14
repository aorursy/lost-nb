import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

data = pd.read_cvs(open("../input/train.cvs",header=0)
                          
print (data)
                  

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


data = pd.read_cvs("../input/train.cvs", header = 0)

