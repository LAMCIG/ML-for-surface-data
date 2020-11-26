#import numpy as np
import sys
from concon_utils import load_raw_labels_boris
#from concon_utils import load_mesh_boris
gs_point = sys.argv[2]    # For an intance:  0.03_0.5
gs_coef = load_raw_labels_boris(gs_point + '_gs_coef.raw')
subject_loc = sys.argv[1]
n=0
#for region in ['11']:
for region in ['10', '11', '12', '13', '17', '18', '26', '49', '50', '51', '52', '53', '54', '58']:
    region_id = subject_loc + '/LogJacs_' + region + '.raw'
    region_size = len(load_raw_labels_boris(region_id))
    gs_coef[n:(n + region_size - 1)].tofile(region + '_' + 'gs_coef.raw')
    n = n + region_size
    #print(n)
