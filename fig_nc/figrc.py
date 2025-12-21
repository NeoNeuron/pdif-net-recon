import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
@FuncFormatter
def sci_formatter(x, pos):
    return r'$10^{%d}$'%x

def add_log_minor_ticks(ax, vrange, where='x'):
    ticks_minor = [np.arange(1,10)*10**i for i in range(*vrange)]
    ticks_minor = np.log10(np.asarray(ticks_minor).flatten())
    if where == 'x':
        ax.set_xticks(ticks_minor, minor=True)
    elif where == 'y':
        ax.set_yticks(ticks_minor, minor=True)
    return ax

def make_graph_diagram(G, ax, pos,
    node_size=2000, font_size=26,
    arrow_size=20, arrow_width=2,
    node_color='#F49227'):
    nx.draw_networkx_nodes(
        G, pos=pos, ax=ax,
        node_color=node_color,
        edgecolors='k',
        node_size=node_size,
    )
    nx.draw_networkx_labels(
        G, pos=pos, ax=ax,
        labels={n:n for n in G},
        font_size=font_size,
        font_weight='bold',
    )
    nx.draw_networkx_edges(
        G, pos=pos, ax=ax,
        width=arrow_width,
        node_size=node_size,
        arrowsize=arrow_size,
        arrows=True,
    )
    ax.set_clip_on(False)
    ax.axis('equal')
    ax.axis('off')


class zoomedAxes(object):
    def __init__(self, ax, zoom_xrange, zoom_yrange, inset_anchor):
        self.ax = ax
        self.zax = self.ax.inset_axes(inset_anchor)
        self.zax.set_xlim(*zoom_xrange)
        self.zax.set_ylim(*zoom_yrange)
        self.zax.spines['top'].set_visible(True)
        self.zax.spines['right'].set_visible(True)
        self.rect = plt.Rectangle(
            (zoom_xrange[0], zoom_yrange[0]),
            zoom_xrange[1] - zoom_xrange[0],
            zoom_yrange[1] - zoom_yrange[0],
            edgecolor='#777777', facecolor='none', linestyle='--')
        self.ax.add_patch(self.rect)


    def plot(self, *args, **kwargs):
        self.ax.plot(*args, **kwargs)
        self.zax.plot(*args, **kwargs)

    def axhline(self, *args, **kwargs):
        self.ax.axhline(*args, **kwargs)
        self.zax.axhline(*args, **kwargs)


from scipy.optimize import curve_fit
def linearfit(x, y):
    def func(x, a):
        return a*x
    popt, _ = curve_fit(func, x, y)
    return lambda x: func(x, *popt)

def squarefit(x, y):
    def func(x, a):
        return a*x**2
    popt,_ = curve_fit(func, x, y)
    return lambda x: func(x, *popt)