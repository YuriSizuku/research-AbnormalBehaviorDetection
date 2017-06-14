from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
import numpy as np
aa = np.array([[1, 2, 3],[3,4,5],[6,7,8]])
bb =  aa.T + aa.T + aa.T
print(aa)
print(bb)