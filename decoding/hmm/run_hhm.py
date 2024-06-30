import numpy as np
from pathlib import Path
from glhmm import glhmm, preproc, prediction, io, graphics

def prep_data(sessions, path):
    all_X = []
    all_y = []
    indices = [] # to keep track of the indices of concatenation


    ind = 0
    for i, session in enumerate(sessions):
        X = []
        y = []

        for sess in session:
            X_tmp = np.load(path / f"{sess}-parcelled.npy")
            X.append(X_tmp)
            start_ind = ind

            ind += X_tmp.shape[1]

            end_ind = ind
            indices.append([start_ind, end_ind])
            
            # get the y
            y_tmp = np.repeat(session_days[i], X_tmp.shape[1])
            y.append(y_tmp)


        all_X.append(np.concatenate(X, axis=1))
        all_y.append(np.concatenate(y))

    X = np.concatenate(all_X, axis=1)
    y = np.concatenate(all_y, axis=0)

    indices = np.array(indices)

    return X, y, indices


if __name__ in "__main__":
    sessions = [['visual_03', 'visual_04'], ['visual_05', 'visual_06', 'visual_07'], ['visual_08', 'visual_09', 'visual_10'], ['visual_11', 'visual_12', 'visual_13'],['visual_14', 'visual_15', 'visual_16', 'visual_17', 'visual_18', 'visual_19'],['visual_23', 'visual_24', 'visual_25', 'visual_26', 'visual_27', 'visual_28', 'visual_29'],['visual_30', 'visual_31', 'visual_32', 'visual_33', 'visual_34', 'visual_35', 'visual_36', 'visual_37', 'visual_38']]
    session_days = [0, 1, 7, 8, 14, 21, 35, 36, 145, 159, 161]

    X_path = Path("/media/8.1/final_data/laurap/rest_continuous_for_hmm")

    X, y, inds = prep_data(sessions, X_path)

    # reshape 
    X = X.reshape(X.shape[1], X.shape[0])

    # only keep two parcels for testing
    X = X[:, :2]

    # preprocess the data
    X, y = preproc.preprocess_data(X, inds)

    hhm = glhmm.glhmm(K = 5, model_beta="no")

    hhm.train(X = None, Y = X, indices = inds)