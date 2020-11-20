import numpy as np
import librosa
import matplotlib
from tqdm import tqdm

def dtw2d(ts_x, ts_y, metrics=lambda x,y: abs(x-y),window=0):
    """
    ts_x, ts_y: Timeseries data. this must be 2d-array (2d tensor)
    metrics: Lambda function. Default is Euclidian distance (abs(x-y)).
    """

    if not (type(ts_x)==np.ndarray and type(ts_y)==np.ndarray):
        raise("type of ts_x and ts_y must nparray")

    if not ts_x.shape[0] == ts_y.shape[0]:
        raise("ts_x[0] and ts_y[0] must be same !")

    ts_x_len = ts_x.shape[1]
    ts_y_len = ts_y.shape[1]

    ts_freq_len = ts_x.shape[0]

    cost_matrix = np.empty((ts_x_len, ts_y_len))
    dist_matrix = np.empty((ts_x_len, ts_y_len))
    

    for n_x in tqdm(range(ts_x_len)):
        ts_x_n = np.reshape(ts_x[:,n_x],[ts_freq_len,1])
        cost_matrix[n_x,:] = np.sum(np.abs(ts_y - ts_x_n), axis=0)

    ## calculate dist matrix
    dist_matrix[0][0] = cost_matrix[0][0]

    for i in range(1,ts_x_len):
        dist_matrix[i][0] = dist_matrix[i-1, 0] + cost_matrix[i, 0]
    
    for j in range(1,ts_y_len):
        dist_matrix[0][j] = dist_matrix[0, j-1] + cost_matrix[0, j]
    
    for i in range(1, ts_x_len):
        windowstart = max(1, i-window)
        windowend = min(ts_y_len, i+window)
        for j in range(windowstart, windowend):
            dist_matrix[i][j] = min(dist_matrix[i-1][j], dist_matrix[i][j-1], dist_matrix[i-1][j-1]) + cost_matrix[i][j]

    
    return dist_matrix[ts_x_len-1][ts_y_len-1]


