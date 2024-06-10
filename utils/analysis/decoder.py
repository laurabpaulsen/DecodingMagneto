import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

class Decoder():
    def __init__(self, alpha:str or float, ncv:int, model_type:str = "LDA", classification:bool = True, get_tgm:bool = True, verbose:bool = True, return_betas:bool = False):
        """
        Parameters
        ----------
        alpha : str or float
            alpha level for LDA or RidgeClassifier
        ncv : int
            number of cross validation folds
        model_type : str, optional
            model type, by default "LDA" (Linear Discriminant Analysis). Currently only LDA and RidgeClassifier are supported
        classification : bool, optional
            classification or regression, by default True
        get_tgm : bool, optional
            get time generalization matrix, by default True
        verbose : bool, optional
            print progress, by default True
        """
        self.classification = classification
        self.alpha = alpha
        self.ncv = ncv
        self.model_type = model_type
        self.get_tgm = get_tgm
        self.verbose = verbose
        self.return_betas = return_betas


    def check_y_format(self,y):
        y = np.copy(y)
        y = y * 1 # convert to int if it was Boolean 
        if not self.classification:
            y = y.astype(float)
            return y

        y = y.astype(int)
        values = np.unique(y)
        ycopy = np.copy(y)
        for k in range(len(values)):
            y[ycopy == values[k]] = k+1

        return y
    
    def _return_pipeline(self):
        if self.model_type == 'LDA':
            model = make_pipeline(StandardScaler(), LDA(solver = 'lsqr', shrinkage = self.alpha))
        elif self.model_type == 'RidgeClassifier':
            model = make_pipeline(StandardScaler(), RidgeClassifier(solver = 'lsqr'), shrinkage = self.alpha)
        else:
            raise ValueError('Model type not supported')
        
        return model
    
    def empty_accuracy_array(self, T):
        if self.get_tgm:
            scores = np.zeros((T, T, self.ncv))
        elif not self.get_tgm:
            scores = np.zeros((T, self.ncv))
        
        return scores

    def run_decoding(self, X, y):
        T, N, C  = X.shape # T = time, N = trials, C = channels/parcels
        y = self.check_y_format(y)

        # making array with all the indices of y for cross validation
        inds = np.array(range(N))
        np.random.shuffle(inds)

        scores = self.empty_accuracy_array(T)

        if self.return_betas:
            betas = np.zeros((T, C, self.ncv))

        for c in range(self.ncv):
            if self.verbose:
                print('Cross validation: ', c+1, '/', self.ncv)
            inds_cv_test = inds[int(len(inds)/self.ncv) * c : int(len(inds)/self.ncv)*(c+1)]

            X_test = X[:, inds_cv_test, :]
            X_train = np.delete(X, inds_cv_test, axis=1)
            y_test = y[inds_cv_test]
            y_train = np.delete(y, inds_cv_test)


            for t in range(T):
                X_t = X_train[t, :, :]
                
                model = self._return_pipeline()

                model.fit(X_t, y_train)

                # add beta values to betas array
                if self.return_betas:
                    if self.model_type == "LDA":
                        betas[t, :, c] = model.named_steps['lineardiscriminantanalysis'].coef_
                    elif self.model_type == "RidgeClassifier":
                        betas[t, :, c] = model.named_steps['ridgeclassifier'].coef_


                if self.get_tgm:
                    for t2 in range(T):
                        X_t2 = X_test[t2, :, :]
                        scores[t, t2, c] = model.score(X_t2, y_test)

                
                elif not self.get_tgm:
                    X_t2 = X_test[t, :, :]
                    scores[t, c] = model.score(X_t2, y_test)

            if self.get_tgm:
                accuracies = np.mean(scores, axis = 2)

            elif not self.get_tgm:
                accuracies = np.mean(scores, axis = 1)


        return accuracies


    def run_decoding_across_sessions(self, X_train, y_train, X_test, y_test):
        T, N_train, C  = X_train.shape # T = time, N = trials, C = channels
        T, N_test, C = X_test.shape # T = time, N = trials, C = channels


        y_train = self.check_y_format(y_train)
        y_test = self.check_y_format(y_test)

        inds_train = np.array(range(N_train))
        inds_test = np.array(range(N_test))
        np.random.shuffle(inds_train)
        np.random.shuffle(inds_test)

        scores = self.empty_accuracy_array(T)

        if self.return_betas:
            betas = np.zeros((T, C, self.ncv))

        for c in range(self.ncv):
            if self.verbose:
                print('Cross validation: ', c+1, '/', self.ncv)
            inds_tmp_train = inds_train[:]
            inds_tmp_train = np.delete(inds_tmp_train, slice(int(len(inds_tmp_train)/self.ncv) * c, int(len(inds_tmp_train)/self.ncv)*(c+1)))
            
            inds_tmp_test = inds_test[int(len(inds_test)/self.ncv) * c : int(len(inds_test)/self.ncv)*(c+1)]
        

            X_train_tmp = np.delete(X_train, inds_tmp_train, axis=1)
            y_train_tmp = np.delete(y_train, inds_tmp_train)

            X_test_tmp = X_test[:, inds_tmp_test, :]
            y_test_tmp = y_test[inds_tmp_test]

            for t in range(T):
                X_t = X_train_tmp[t, :, :]

                model = self._return_pipeline()

                model.fit(X_t, y_train_tmp)

                # add beta values to betas array
                if self.return_betas:
                    if self.model_type == "LDA":
                        betas[t, :, c] = model.named_steps['lineardiscriminantanalysis'].coef_
                    elif self.model_type == "RidgeClassifier":
                        betas[t, :, c] = model.named_steps['ridgeclassifier'].coef_

                if self.get_tgm:
                    for t2 in range(T):
                        X_t2 = X_test_tmp[t2, :, :]
                        scores[t, t2, c] = model.score(X_t2, y_test_tmp)
                

                elif not self.get_tgm:
                    X_t2 = X_test_tmp[t, :, :]
                    scores[t, c] = model.score(X_t2, y_test_tmp)
                    
            if self.get_tgm:        
                accuracies = np.mean(scores, axis = 2)

            elif not self.get_tgm:
                accuracies = np.mean(scores, axis = 1)

        return accuracies