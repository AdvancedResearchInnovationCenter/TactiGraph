import numpy as np
from matplotlib import pyplot as plt
from IPython import get_ipython


def events_plot_3d(events_data):
    events_data[:,2] = (events_data[:,2] - events_data[0,2]) / 1e9
    events_positive = events_data[events_data[:,3]==1, 0:3]
    events_negative = events_data[events_data[:,3]==0, 0:3]

    # Visualize events in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    px_pos = events_positive[:,0]
    py_pos = events_positive[:,1]
    t_st_pos = events_positive[:,2]
    ax.scatter(px_pos, py_pos, t_st_pos, c="red", s=5, marker='.')
    px_neg = events_negative[:,0]
    py_neg = events_negative[:,1]
    t_st_neg = events_negative[:,2]
    ax.scatter(px_neg, py_neg, t_st_neg, c="blue", s=5, marker='.')
    plt.show()
    
    
import seaborn as sns
def plot(ev):
    ev = np.array(ev)
    plt.figure(figsize=(5, 5))
    print(ev.shape)
    
    sns.scatterplot( x=ev[:, 0],y=ev[:, 1], hue=ev[:, 3])
    # sns.scatterplot(x=ev[:, 0], y=ev[:, 1], hue=ev[:, 3])
