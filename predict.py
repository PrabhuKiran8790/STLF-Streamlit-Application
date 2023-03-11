import numpy as np
import pandas as pd
import tensorflow as tf
import joblib as jbl

def scale(x, min, max):
    return np.array([[(x[i] - min[i]) / (max[i] - min[i]) for i in range(len(x))]])

def descale(x_scaled, min, max):
    return (max - min) * x_scaled + min

def out_matrix(X, n_hidden, centroids, sigma_mat):
        out_mat = np.zeros((X.shape[0], n_hidden))
        for i in range(len(X)):
            for j in range(n_hidden):
                numerator = -np.square(X[i] - centroids[j]).sum()
                denominator = 2*(np.square(sigma_mat[j]))
                out_mat[i][j] = np.exp(numerator/denominator)
        return out_mat

def autoEncoder(inputs, file_path:str):
    ae_model = tf.keras.models.load_model(file_path)
    return pd.DataFrame(ae_model.predict(inputs)).drop(3, axis=1).to_numpy()

def predict(inputs, file_path:str, ae_filepath:str=None, load=False):
    metadata = jbl.load(file_path)  
    if load:
        inputs = scale(inputs, metadata['x_min'], metadata['x_max'])
        x_scaled = autoEncoder(inputs.reshape((1, 12)), ae_filepath)      
    else: x_scaled = scale(inputs, metadata['x_min'], metadata['x_max'])
    out_mat = out_matrix(x_scaled, metadata['neurons'], metadata['centers'], metadata['sigma'])
    print(out_mat)
    ann_out = out_mat @ metadata['ann_weights'] + metadata['ann_bias']
    return descale(ann_out, metadata['y_min'], metadata['y_max'])