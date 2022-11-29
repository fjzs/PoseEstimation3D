# In this file we run the evaluation phase on the test set
import numpy as np
import paths
import utils


# 1: Get the split files
file_names, rgb, depth, label, meta = utils.get_split_files("test")
N = 10
file_names = file_names[0:N]
rgb = rgb[0:N]
depth = depth[0:N]
label = label[0:N]
meta = meta[0:N]

print(file_names)



