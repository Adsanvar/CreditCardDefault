import pandas as pd
import numpy as np


data = pd.read_excel('default_of_credit_card_clients.xls')
data = np.array(data)
data = data[1:,:-1]

print(data)