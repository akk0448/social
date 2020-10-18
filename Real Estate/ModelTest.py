# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 01:22:14 2020

@author: Aniket
"""

import numpy as np
from joblib import dump, load
regressor = load('housing.joblib')

input = np.array([[-0.408359, -0.499608, -1.12873, -0.272888, -0.833369, 0.044972, -1.84622, 0.695069, -0.624648, 0.159137, -0.712729, 0.185476, -0.736103]])
x = (int)(regressor.predict(input)*1000)
print("Predicted Cost = $",x)


