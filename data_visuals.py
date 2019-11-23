

import matplotlib.pyplot as plt
import matplotlib.style as style

style.use('seaborn-poster') #sets the size of the charts
style.use('ggplot')

def plot_time_avg(tick_times, times, values, lx, ly, title):

    plt.xticks(tick_times)
    plt.xticks(rotation=45)
    plt.plot(times, values, linestyle=':')

    plt.title(title)
    plt.xlabel(lx)
    plt.ylabel(ly)

    plt.show()

def plot_freq(dict, title):
    ax = plt.axes()
    plt.setp(ax.get_xticklabels(), rotation=90, horizontalalignment='right', fontsize='x-small')
    plt.bar(dict.keys(), dict.values(), 0.75, color='b')

    plt.title(title)
    plt.xlabel('times')
    plt.ylabel('frequency')
    plt.tight_layout()

    plt.show()