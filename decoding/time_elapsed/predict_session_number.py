from pathlib import Path
import argparse as ap
import numpy as np
import json
import itertools
from scipy.stats import spearmanr
import pickle
from multiprocessing import Pool

# local imports
import sys
sys.path.append(str(Path(__file__).parents[2])) # adds the parent directory to the path so that the utils module can be imported
from utils.data.concatenate import read_and_concate_sessions
from utils.data.triggers import balance_class_weights_multiple
from ridge_fns import tgm_ridge_scores

def parse_args():
    parser = ap.ArgumentParser()
    parser.add_argument('--trial_type', type=str, default='animate', help='Trial type. Can be either animate or inanimate.')
    parser.add_argument('--task', type=str, default='visual', help='Task. Can be either visual or memory or visualsubset.')
    parser.add_argument('--ncv', type=int, default=10, help='Number of cross validation folds.')
    parser.add_argument('--nperms', type=int, default=24, help='Number of permutations to use.')

    args = parser.parse_args()
    
    return args

def prepare_data(args):
    session_list, session_days, triggers = args
    """
    Prepares data for decoding by concatenating sessions from different days. The y array is the session number.

    Parameters
    ----------
    session_list : list
        List of lists of sessions to be concatenated.
    session_days : list
        List of session days. That is, the day of the session relative to the first session.
    triggers : list
        List of triggers to be used.

    Returns
    -------
    X : np.array
        Data array.
    y : np.array
        Label array
    """
    
    for i, sesh in enumerate(session_list):
        sesh = [f'{i}-epo.fif' for i in sesh]
        X, y = read_and_concate_sessions(sesh, triggers)

        # replace all values of y with day of the session as found in session_days
        y = np.array([session_days[i] for _ in range(len(y))])

        if i == 0:
            Xs = X
            ys = y
        else:
            Xs = np.concatenate((Xs, X), axis=1)
            ys = np.concatenate((ys, y), axis=0)

    # balance class weights
    X, y = balance_class_weights_multiple(Xs, ys)

    return X, y

def get_triggers(trial_type:str = "animate"):
    path = Path(__file__)

    with open(path.parents[2] / 'info_files' / 'event_ids.txt', 'r') as f:
        file = f.read()
        event_ids = json.loads(file)

    triggers = [value for key, value in event_ids.items() if trial_type.capitalize() in key][:27]

    return triggers

def run_permutation(args):
    permute_session_days, session_days, sessions, triggers, args_ncv = args
    X, y = prepare_data((sessions, permute_session_days, triggers))
    pred, true = tgm_ridge_scores(X, y, cv=5, ncv=args_ncv)
    return {
        "permuted_session_days": permute_session_days,
        "true_session_days": session_days,
        "correlation": spearmanr(session_days, permute_session_days).correlation,
        "predicted": pred, 
        "true": true
    }

if __name__ == '__main__':
    args = parse_args()

    path = Path(__file__)
    output_path = path.parent / "results" / f'{args.trial_type}_{args.task}_session_number.pkl'

    # ensure that the results directory exists
    output_path.parents[0].mkdir(parents=True, exist_ok=True)
    
    sessions = [['visual_03', 'visual_04'], ['visual_05', 'visual_06', 'visual_07'], ['visual_08', 'visual_09', 'visual_10'], ['visual_11', 'visual_12', 'visual_13'], ['memory_01', 'memory_02'], ['memory_03', 'memory_04', 'memory_05', 'memory_06'],  ['memory_07', 'memory_08', 'memory_09', 'memory_10', 'memory_11'], ['memory_12', 'memory_13', 'memory_14', 'memory_15'] ,['visual_14', 'visual_15', 'visual_16', 'visual_17', 'visual_18', 'visual_19'],['visual_23', 'visual_24', 'visual_25', 'visual_26', 'visual_27', 'visual_28', 'visual_29'],['visual_30', 'visual_31', 'visual_32', 'visual_33', 'visual_34', 'visual_35', 'visual_36', 'visual_37', 'visual_38']]
    session_days = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    #  change sessions depending on the task
    if args.task == 'visual':
        sessions = sessions[:4] + sessions[8:]
        session_days = session_days[:4] + session_days[8:]

    elif args.task == 'visualsubset':
        # only take the first 4 visual sessions
        sessions = sessions[:4]
        session_days = session_days[:4]

    elif args.task == 'visualsubset_easy':
        # take session with the following indices: 0, 3, 8, 10
        sessions = [sessions[i] for i in [0, 3, 8, 10]]
        session_days = [session_days[i] for i in [0, 3, 8, 10]]

    elif args.task == 'memory':
        sessions = sessions[4:8]
        session_days = session_days[4:8]
    
    else: 
        raise ValueError("Task must be either visual, visualsubset, visualsubset_easy or memory")

    # get the triggers
    triggers = get_triggers(args.trial_type)

    # make a list of all possible permutations of the session list
    session_list_permutations = list(itertools.permutations(session_days))

    # take the permutations most different from the original session list (spearman correlation between original and permuted) closest to 0
    spearman_correlations = [spearmanr(session_days, i).correlation for i in session_list_permutations]

    # get the indices of the ten most different permutations
    indices = np.argsort(np.abs(spearman_correlations))
    
    permutations = [session_list_permutations[i] for i in indices[:args.nperms]]
    corr = [spearman_correlations[i] for i in indices[:args.nperms]]

    output = {}

    args_list = [(permute_session_days, session_days, sessions, triggers, args.ncv) for permute_session_days in permutations]
    
    # Using multiprocessing to parallelize
    with Pool(1) as p:
        results = p.map(run_permutation, args_list)

    for i, res in enumerate(results):
        output[i+1] = res

    # also run the original data
    X, y = prepare_data((sessions, session_days, triggers))
    pred, true = tgm_ridge_scores(X, y, cv=5, ncv=args.ncv)
    output["original"] = {
        "predicted": pred, 
        "true": true
    }

    with open(output_path, 'wb') as f:
        pickle.dump(output, f)
