# usage: python3 heatmap_kl.py npy_fname

import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

npy_fname = sys.argv[1]
heatmap_fname = npy_fname.replace('.npy', '.png')
print('produce %s...' % heatmap_fname)

class_list = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_class = len(class_list)

data = np.load(npy_fname)

# normalize each row
row_sums = data.sum(axis=1, keepdims=True)
data = data / row_sums

# print('data:')
# for i in range(num_class):
#     for j in range(num_class):
#         print('%.4f' % data[i][j], end=' ')
#     print()

# create plot
fig, ax = plt.subplots(figsize=(8,8))
im = ax.imshow(data, cmap='Blues', vmin=0.0, vmax=1.0)

# set axes
ax.set_xticks(np.arange(len(class_list)))
ax.set_yticks(np.arange(len(class_list)))
# Y-axis will be cut-off on both top and bottom, need the following workaround
# Ref: https://github.com/matplotlib/matplotlib/issues/14751
ax.set_ylim(len(class_list)-0.5, -0.5)
ax.set_xticklabels(class_list, rotation=30)
ax.set_yticklabels(class_list)
ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

# add colorbar
from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.1)
plt.colorbar(im, cax=cax)

# add text in each cell
textcolors=('black', 'white')
# threshold = im.norm(data.max())/2.
threshold = 0.5
kw = dict(horizontalalignment='center', verticalalignment='center')
valfmt = matplotlib.ticker.StrMethodFormatter('{x:.3f}')
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
        text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)

# save
plt.savefig(heatmap_fname)
