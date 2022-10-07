import pickle

def load_data():
 
    filename = '/home/yichen/ts2vec/datafiles/Geolife/traindata_4class_xy_traintest_interpolated_trip%d.pickle' % 5

    filename = '/home/yichen/ts2vec/datafiles/Geolife/traindata_4class_xy_traintest_trip%d.pickle ' % 5
    
    with open(filename, 'rb') as f:
        kfold_dataset, X_unlabeled = pickle.load(f)
    dataset = kfold_dataset

    print('dataset:', dataset)

load_data ()