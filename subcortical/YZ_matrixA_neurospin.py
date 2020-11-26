import numpy as np
import parsimony.functions.nesterov.tv as tv_helper
from concon_utils import load_mesh_boris
## mesh structure ##
def A_main():
    mesh_coord = []
    mesh_triangles = []
    #for region in ['18']:
    for region in ['10', '11', '12', '13', '17', '18', '26', '49', '50', '51', '52', '53', '54', '58']:
        #atlas_loc = '/home/yzhao104/Desktop/BG_project/atlas'
        atlas_loc = '.'
        region_id = 'atlas_{}'.format(region) + '.m'
        #print(region_id)
        vertex, face  = load_mesh_boris(atlas_loc + '/' + region_id)
        #print(vertex.shape, face.shape)
        mesh_coord.append(list(vertex))
        mesh_triangles.append(list(face.astype(np.int16)))
    #print(len(mesh_coord))
    #print(mesh_coord[13][1])
    mesh_coord = np.vstack(mesh_coord)
    #print('mesh_coord: {}'.format(mesh_coord.shape))
    mesh_triangles = np.vstack(mesh_triangles)
    #print('mesh_triangles: {}'.format(mesh_triangles.shape))

    A = tv_helper.linear_operator_from_mesh(mesh_coord, mesh_triangles)
    return A
