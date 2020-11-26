import numpy as np
from concon_utils import load_raw_labels_boris
import sys
gs_coef = load_raw_labels_boris(sys.argv[1] + '_gs_coef.raw')
print('coef range: ({},{})'.format(gs_coef.min(), gs_coef.max()))
display = max(np.absolute(gs_coef.min()), np.absolute(gs_coef.max()))
print('display_lb:{}'.format(-display))
print('display_ub:{}'.format(display))
dead = np.where((gs_coef <0.00001) & (gs_coef > -0.00001))[0].shape[0]  # np.where() is data-mining-wise function of numpy library
print('dead(-0.00001,+0.00001) coefs {}%'.format(int(100*float(dead)/len(gs_coef))))
