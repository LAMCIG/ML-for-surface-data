import numpy as np
#import pandas as pd
import os

#from nilearn.surface import load_surf_mesh, load_surf_data

#from nibabel.freesurfer.io import read_geometry
#from sklearn.neighbors import KNeighborsClassifier

def load_mesh_boris(path='/home/bgutman/datasets/HCP/Dan_iso5.m'):

    '''
    load boris mesh (.m) file
    faces enumerated from 1, but after loading from 0


    usage:
    vertices, faces = load_mesh_boris('/home/bgutman/datasets/HCP/Dan_iso5.m')
    '''
    with open(path, 'r') as f:
        iso5 = f.read()
    iso5 = iso5.split('\n')
    vertices = []
    faces = []
    #ind=[]
    for line in iso5:
        a = line.split(' ')
        if a[0] == 'Vertex':
            vertices.append([float(sym) for sym in a[2:5]])
        elif a[0] == 'Face':
            faces.append([int(sym) for sym in a[2:]])
            #ind.append(int(a[1]))
    vertices = np.array(vertices)
    faces = np.array(faces) - 1
    return vertices, faces#, ind

def load_raw_labels_boris(path):
    '''
    load boris labels from .raw format
    labels = load_raw_labels_boris('LH_labels_MajVote.raw')
    '''
    with open(path, 'rb') as f:
        labels = np.fromfile(f, count=-1 ,dtype='float32')
    return labels


def squeeze_matrix(matrix, labels, return_transform=False, fill_diag_zero=True):
    '''
    given a matrix,
    and indexes
    sum appropriate rows and columns
    of a matrix, return another matrix
    of a different size

    EXAMPLE :

    labels = np.array([0,0,1,1,2,2])
    idexes = np.array([[row, col] for row,col in enumerate(labels)])
    transform = np.zeros((6, 3))
    for index in idexes:
        transform[index[0], index[1]] = 1

    A = np.array([[0, 4, 5, 1, 5, 4],
                  [4, 0, 3, 4, 2, 5],
                  [5, 3, 0, 5, 8, 5],
                  [1, 4, 5, 0, 4, 4],
                  [5, 2, 8, 4, 0, 6],
                  [4, 5, 5, 4, 6, 0]])

    transform = np.array(
            [[1,0,0],
             [1,0,0],
             [0,1,0],
             [0,1,0],
             [0,0,1],
             [0,0,1]])

    squeezed = transform.T.dot(A.dot(transform))
    np.fill_diagonal(squeezed, 0)

    squeezed
    array([[ 0, 13, 16],
           [13, 0, 21],
           [16, 21, 0]])
    '''
    input_size = matrix.shape[0]
    output_size = np.unique(labels).shape[0]

    d = dict(zip(np.unique(labels), np.arange(output_size)))
    _labels = list(map(lambda x: d[x], labels))
    transform = np.zeros((input_size, output_size))

    for row, col in enumerate(_labels):
        transform[row, col] = 1

    squeezed = transform.T.dot(matrix.dot(transform))

    if fill_diag_zero:
        np.fill_diagonal(squeezed, 0)

    if return_transform:
        return squeezed, transform

    return squeezed


def read_FS_stats(path):
    '''
    This script parse FreeSurfer .stats files
    particulary [lh,rh].aparc.DKTatlas40.stats into pandas table

    TODO:
    manually check whether it correctly deals with other .stats files
    works well for .aparc.a2009s.stats, .aparc.stats, .aparc.DKTatlas40.stats

    tips:

    last row that starts with # stands for column names,
    all previous rows are meta info
    '''

    def parse_row(row):
        '''
        Parse single row
        '''
        row_data = []
        row_array = row.split(' ')
        row_data.append(row_array[0])
        for elem in row_array[1:]:
            try:
                num = float(elem)
                row_data.append(num)
            except:
                pass

        return row_data

    with open(path, 'r') as f:
            stats_file = f.read()

    _rows = stats_file.split('\n')
    rows = [row for row in _rows if row != '']
#    n_rows = len(rows)

    header_row = 0
    _data = []
    for row in rows:
        if row[0] == '#':
            header_row += 1
        else:
            _data.append(row)

    header = rows[header_row - 1]
    header = header.split(' ')[2:]

    data = []
    for row in _data:
        data.append(parse_row(row))

    stats = pd.DataFrame(data = data, columns=header)

    return stats


def downsample_mesh_faces(faces, source='IC7', destination='IC6'):
    d = {'IC7':[40962, 163842], 'IC6':[10242, 40962], 'IC5':[2562, 10242], 'IC4': [642, 2562], 'IC3' : [162, 642]}
    from tqdm import tqdm_notebook
    start = d[source][0]
    end = d[source][1]

    def get_ind(faces, vertice, start):
        '''
        returns 2 vertices from lower resolution
        and 2 couple of vertices from same resolution

        a, b = get_ind(faces, 163841)
        print(a, b)
        [39990 40719] [array([163840,  92160]), array([163113, 160926])]

        '''
        low_level_vertices = []
        high_level_vertices = []
        for triag in faces[np.where(faces == vertice)[0]]:
            if np.any(triag<start):#40962):
                low_level_vertices.append(triag.min())
            elif np.all(triag>=start):#40962):
                high_level_vertices.append(triag[triag!=vertice])

        return np.unique(low_level_vertices), high_level_vertices


    def get_third_vertice(faces, high_level_vertices):
        res = []
        for vertices in high_level_vertices:
            indexes = np.intersect1d(np.where(faces == vertices[0])[0], np.where(faces == vertices[1])[0])
            res.append(faces[indexes].min())

        return res[0], res[1]

    new_faces = []



    for vertice in tqdm_notebook(range(start, end)):#40962, 163842)):
        low_level_vertices, high_level_vertices = get_ind(faces, vertice, start)
        v1, v2 = low_level_vertices[0], low_level_vertices[1]
        v3, v4 = get_third_vertice(faces, high_level_vertices)

        new_faces.append(sorted([v1, v2, v3]))
        new_faces.append(sorted([v1, v2, v4]))

    new_faces = np.array(new_faces)
    import pandas as pd
    triags = pd.DataFrame(data=new_faces, )
    #triags = triags.sort_values(by=0)

    return triags.drop_duplicates().values




def transfer_mesh_color(source_mesh_path, source_mesh_color_path, destination_mesh_path, n_neighbors=1):
    '''
    Given Boris sphere/brain-like mesh and FreeSurfer sphere/brain-like mesh
    and colors for one of them, transfer that color to another.

    Boris uses .m for meshes, .raw for labels
    FreeSurfer use different formats amongst other:
    '''

    if '.m' in source_mesh_path and '.raw' in source_mesh_color_path:
        s_vertices, s_faces = load_mesh_boris(source_mesh_path)
        s_labels = load_raw_labels_boris(source_mesh_color_path)
        s_vertices *= 100 # Both Boris and FS use unit sphere but FS multiply all coordinate by 100

        d_vertices, d_faces = read_geometry(destination_mesh_path) # load_surf_mesh(destination_mesh_path)
    else:
        s_vertices, s_faces = read_geometry(source_mesh_path) # load_surf_mesh(source_mesh_path)
        s_labels = load_surf_data(source_mesh_color_path)

        d_vertices, d_faces = load_mesh_boris(destination_mesh_path)
        d_vertices *= 100


    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(s_vertices, s_labels)

    d_labels = knn.predict(d_vertices)

    return d_labels
