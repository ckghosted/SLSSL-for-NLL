# usage: python3 heatmap_noise.py noise_mode noise_ratio
#        noise_mode: sym/asym/unnat
#        noise_ratio: 0.1/0.2/...

import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

noise_mode = sys.argv[1]
noise_ratio = float(sys.argv[2])
heatmap_fname = noise_mode + ''.join(str(noise_ratio).split('.')) + '_heatmap.png'
print('produce %s...' % heatmap_fname)

class_list = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_class = len(class_list)

if noise_mode == 'sym':
    data = np.full((num_class, num_class), fill_value=noise_ratio/num_class)
    for i in range(num_class):
        data[i][i] = data[i][i] + (1-noise_ratio)
else:
    if noise_mode == 'asym':
        transition = {0:0,2:0,4:7,7:7,1:1,9:1,3:5,5:3,6:6,8:8}
    elif noise_mode == 'unnat':
        transition = {0:7,1:1,2:2,3:1,4:4,5:5,6:5,7:0,8:2,9:9}
    data = np.full((num_class, num_class), fill_value=0.0)
    for i in range(num_class):
        data[i][i] = data[i][i] + (1-noise_ratio)
        data[i][transition[i]] = data[i][transition[i]] + noise_ratio

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
