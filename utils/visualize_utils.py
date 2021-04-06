
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

def barplot(x,bins,title):
    y_pos = np.arange(len(bins))
    plt.bar(y_pos, x)
    plt.title(title) 
    plt.xlabel('class'); 
    plt.ylabel('percent')
    plt.xticks(y_pos, bins)
    plt.show()