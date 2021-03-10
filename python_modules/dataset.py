""" load training data
"""


# galaxy images
def get_x_train():
    x_train = np.load(file_galaxy_images)
    x_train = x_train/255.0 ## rescale to 0<x<1
    x_train = np.rollaxis(x_train, -1, 1)  ## pytorch: (colors,dim,dim)
    N_samples = x_train.shape[0]
    x_train = from_numpy(x_train).cuda()
    return x_train

# hierarchical galaxy labels
def get_labels_train():
    df_galaxy_labels =  pd.read_csv(file_galaxy_labels)
    ## for now, only use top level labels
    labels_train = df_galaxy_labels[df_galaxy_labels.columns[1:4]].values
    labels_train = from_numpy(labels_train).float().cuda()
    return labels_train
