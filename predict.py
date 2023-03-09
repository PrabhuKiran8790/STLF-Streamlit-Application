import numpy as np
import tensorflow as tf
import joblib as jbl

def out_matrix(X, n_hidden, centroids, sigma_mat):
        out_mat = np.zeros((X.shape[0], n_hidden))
        for i in range(len(X)):
            for j in range(n_hidden):
                numerator = -np.square(X[i] - centroids[j]).sum()
                denominator = 2*(np.square(sigma_mat[j]))
                out_mat[i][j] = np.exp(numerator/denominator)
        return out_mat


def predict(x, meta_file, model_file):
    metadata = jbl.load(meta_file)
    centers = metadata['centers']
    sigma = metadata['sigma']
    n_hidden = metadata['neurons']
    out_mat = out_matrix(X=x, n_hidden=n_hidden, centroids=centers, sigma_mat=sigma)[-1].reshape((1, n_hidden))
    return tf.keras.models.load_model(model_file, compile=False).predict(out_mat)