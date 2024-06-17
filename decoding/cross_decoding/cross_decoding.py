"""
This script is used to decode both source and sensor space data using cross decoding.
"""

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
import numpy as np
import argparse
from pathlib import Path

# local imports
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parents[2])) # adds the parent directory to the path so that the utils module can be imported
print(str(pathlib.Path(__file__).parents[2]))
from utils.data.concatenate import read_and_concate_sessions
from utils.data.triggers import get_triggers_equal, convert_triggers_animate_inanimate, balance_class_weights
from utils.data.prep_data import equalise_trials
from tqdm import tqdm

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


""" -------- DEFINE FUNCTIONS -------- """

def parse_args():
    parser = argparse.ArgumentParser(description='This script is used to decode both source and sensor space data using cross decoding.')
    parser.add_argument('--model_type', type=str, default='LDA', help='Model type. Can be either LDA or RidgeClassifier.')
    parser.add_argument('--parc', type=str, default='sens', help='Parcellation. Can be either aparc, aparc.a2009s, aparc.DKTatlas, HCPMMP1 or sens.')
    parser.add_argument('--ncv', type=int, default=10, help='Number of cross validation folds.')
    parser.add_argument('--alpha', type=str, default='auto', help='Regularization parameter. Can be either auto or float.')

    args = parser.parse_args()


    return args


def prep_data(sessions, triggers):
    Xs, ys = [], []

    for sesh in sessions:
        sesh = [f'{i}-epo.fif' for i in sesh]
        X, y = read_and_concate_sessions(sesh, triggers)

        # converting triggers to animate and inanimate instead of images
        y = convert_triggers_animate_inanimate(y) 
        
        # balance class weights
        X, y, _ = balance_class_weights(X, y)

        Xs.append(X)
        ys.append(y)
    
    return Xs, ys



def create_cv_folds(Xs:list, ys:list, ncv:int, shuffle:bool = True):
    """
    creates a list of cross validation folds for each scanning day

    Parameters
    ----------
    Xs : list
        list of arrays with shape (T, N, C) where T is the number of time points, N is the number of trials and C is the number of sensors
    ys : list
        list of arrays with shape (N,) where N is the number of trials
    ncv : int
        number of cross validation folds
    shuffle : bool, optional
        shuffle the data before splitting, by default True

    Returns
    -------
    Xs_cv : list
        list of lists of arrays with shape (T, N, C) where T is the number of time points, N is the number of trials and C is the number of sensors
    ys_cv : list
        list of lists of arrays with shape (N,) where N is the number of trials
    """
    Xs_cv = []
    ys_cv = []

    for X, y in zip(Xs, ys):
        T, N, C = X.shape
        inds = np.arange(N)

        if shuffle:
            np.random.shuffle(inds)

        cv_folds = np.array_split(inds, ncv)

        X_cv = [X[:, fold, :] for fold in cv_folds]
        y_cv = [y[fold] for fold in cv_folds]

        Xs_cv.append(X_cv)
        ys_cv.append(y_cv)

    return Xs_cv, ys_cv
    

def main():
    args = parse_args()
    ncv = args.ncv
    alpha = args.alpha

    # defining paths
    path = Path(__file__).parent
    output_path = path / "accuracies" / f'cross_decoding_{ncv}_{args.model_type}_sens.npy'
    print(output_path)
    output_path_betas = path / "betas" / f'cross_decoding_{ncv}_{args.model_type}_sens.npy'

    # define logger
    logger = logging.getLogger(__name__)

    # log info
    logger.info(f'Running cross decoding with the following parameters...')
    logger.info(f'Number of cross validation folds: {ncv}')

    sessions = [['visual_03', 'visual_04'], ['visual_05', 'visual_06', 'visual_07'], ['visual_08', 'visual_09', 'visual_10'], ['visual_11', 'visual_12', 'visual_13'],['visual_14', 'visual_15', 'visual_16', 'visual_17', 'visual_18', 'visual_19'],['visual_23', 'visual_24', 'visual_25', 'visual_26', 'visual_27', 'visual_28', 'visual_29'],['visual_30', 'visual_31', 'visual_32', 'visual_33', 'visual_34', 'visual_35', 'visual_36', 'visual_37', 'visual_38']]

    triggers = get_triggers_equal() # get triggers for equal number of trials per condition (27 animate and 27 inanimate)
    
    # reading and concatenating data
    Xs, ys = prep_data(sessions, triggers)

    # equalising number of trials
    Xs, ys = equalise_trials(Xs, ys)

    Xs_cv, ys_cv = create_cv_folds(Xs, ys, ncv)


    # empty array to store accuracies in
    accuracies = np.zeros((len(Xs), len(Xs), Xs[0].shape[0], Xs[0].shape[0], ncv), dtype=float)
    
    # empty array to store betas in (number of scanning days, number of time points, number of sensors, cv folds)
    betas = np.zeros((len(Xs), Xs[0].shape[0], Xs[0].shape[2], ncv), dtype=float)



    for train_sesh in range(len(Xs_cv)):
        logger.info(f'Training on session {train_sesh+1}')
        for cv_test in tqdm(range(ncv), desc=f'cross-validation fold number'):
            # get all the other indices for training than the one used for testing
            train_inds = np.delete(np.arange(ncv), cv_test)

            # get the training data by concatenating the data from the training indices 
            X_train = np.concatenate([Xs_cv[train_sesh][i] for i in train_inds], axis=1)
            y_train = np.concatenate([ys_cv[train_sesh][i] for i in train_inds])

            # loop over time points
            for t in range(X_train.shape[0]):
                X_train_t = X_train[t, :, :]

                model = make_pipeline(StandardScaler(), LDA(solver = 'lsqr', shrinkage = alpha))

                # fit the model
                model.fit(X_train_t, y_train)

                # get the beta values
                betas[train_sesh, t, :, cv_test] = model.named_steps['lineardiscriminantanalysis'].coef_


                for test_sesh in range(len(Xs)):
                    # get the testing data
                    X_test = Xs_cv[test_sesh][cv_test]
                    y_test = ys_cv[test_sesh][cv_test]

                    # loop over time points
                    for t2 in range(X_test.shape[0]):
                        X_test_t = X_test[t2, :, :]

                        # get the accuracy
                        accuracy = model.score(X_test_t, y_test)

                        accuracies[train_sesh, test_sesh, t, t2, cv_test] = accuracy

    # making sure output path exists
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)

    if not output_path_betas.parent.exists():
        output_path_betas.parent.mkdir(parents=True)
   
    # saving accuracies
    np.save(output_path, accuracies)
    np.save(output_path_betas, betas)

    

if __name__ == '__main__':
    main()
