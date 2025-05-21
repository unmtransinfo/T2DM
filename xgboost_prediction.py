import pickle
import numpy as np
import yaml
import argparse
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, accuracy_score, matthews_corrcoef, f1_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from copy import deepcopy
from sklearn.metrics import confusion_matrix
import random
from scipy.sparse import vstack
from scipy import stats
from sklearn.utils import shuffle
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def get_data_for_ml(features_file=None, labels_file=None, persons_file=None, covars_file=None):
    """
    load saved data for ML method -- features, labels, person_ids and covariate list
    """
    print("\nfetch ML data from the file")
    if features_file is None or labels_file is None or persons_file is None or covars_file is None:
        print("please provide file name for ml data")
        exit(0)
    else:
        with open(features_file, "rb") as fh:
            X_all = pickle.load(fh)

        # labels
        with open(labels_file, "rb") as fh:
            Y_all = pickle.load(fh)

        # persons
        with open(persons_file, "rb") as fh:
            rec_ids_all = pickle.load(fh)

        # covars
        with open(covars_file, "rb") as fh:
            covars_all = pickle.load(fh)

        return X_all, Y_all, rec_ids_all, covars_all

def selected_undiagnosed_using_threshold(X, Y, rec_ids, risk_threshold=1.0, undiag_pred_file=None):
    """
    This function selects undiagnosed patients with risk of T2DM <= given threshold
    """
    print("\nfetch prediction data for undiagnosed patients from the file")
    if undiag_pred_file is None:
        print("please provide file name for prediction data")
        exit(0)
    else:
        # undiagnosed predictions
        with open(undiag_pred_file, 'rb') as f:
            undiag_pred_dict = pickle.load(f)

        # select all pids with risk <= threshold
        undiag_recs = np.asarray([k for k, v in undiag_pred_dict.items() if v[0] <= risk_threshold])
        print(f"number of undiagnosed patients with risk <= {risk_threshold}: {len(undiag_recs)}")

        # select positive and unlabeled data
        X_pos, X_unlab, rec_unlab = X[Y==1], X[Y==0], rec_ids[Y == 0]

        # find the indices of undiag_recs in rec_unlab
        indx = np.isin(rec_unlab, undiag_recs).nonzero()[0]
        X_unlab = X_unlab[indx]

        # create data and label with all positives and selected undiagnosed
        X = vstack([X_pos, X_unlab])  # merge the data
        Y = np.concatenate([[1] * X_pos.shape[0], [0] * X_unlab.shape[0]])

        return X, Y

def run_binary_classifier(X_all, Y_all, rseed=1001, max_itereration=5):
    """
    This function runs XGBoost models to estimate classification performance
    """
    est_acc, est_mcc, est_auc, est_f1, est_aps = [], [], [], [], []
    est_bs, est_sensitivity, est_specificity, est_ppv = [], [], [], []

    # select positive and unlabeled records
    X_pos = X_all[Y_all == 1]
    X_unlab = X_all[Y_all == 0]

    # run for max_iterations
    for itr in range(max_itereration):
        random.seed(rseed*itr)
        indx = random.sample(range(X_unlab.shape[0]), X_unlab.shape[0])  # shuffle indices
        sel_indx = indx[:X_pos.shape[0]]    # same number of data
        sel_X_unlab = X_unlab[sel_indx]
        X = vstack([X_pos, sel_X_unlab])  # combine positive and unlabeled data
        Y = np.concatenate([[1]*X_pos.shape[0], [0] * sel_X_unlab.shape[0]])
        X, Y = shuffle(X, Y, random_state=rseed*itr)
        print(f"\niteration {itr+1}: positive: {X_pos.shape}, unlabeled: {sel_X_unlab.shape}, full data: {X.shape}, {len(Y)}")

        # run model
        model = XGBClassifier(max_depth=4, subsample=0.8, colsample_bytree=0.8, eval_metric='logloss',
                              random_state=rseed*itr)
        cv = StratifiedKFold(n_splits=5, shuffle=False)
        y_pred_proba = cross_val_predict(model, X, Y, cv=cv, method='predict_proba', n_jobs=16) # probability
        y_pred = np.round(y_pred_proba).argmax(axis=1) # label

        # classification performance
        accuracy = accuracy_score(Y, y_pred)
        mcc = matthews_corrcoef(Y, y_pred)
        auc = roc_auc_score(Y, y_pred_proba[:, 1])
        f1_val = f1_score(Y, y_pred)
        aps = average_precision_score(Y, y_pred_proba[:, 1])
        bs = brier_score_loss(Y, y_pred_proba[:, 1])
        tn, fp, fn, tp = confusion_matrix(Y, y_pred).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        ppv = tp / (tp + fp)
        # show results
        print("Accuracy: ", accuracy)
        print("MCC: ", mcc)
        print("AUC: ", auc)
        print("F1: ", f1_val)
        print("APS: ", aps)
        print("BS: ", bs)
        print("Sensitivity: ", sensitivity)
        print("Specificity: ", specificity)
        print("PPV: ", ppv)
        # save values
        est_acc.append(accuracy)
        est_mcc.append(mcc)
        est_auc.append(auc)
        est_f1.append(f1_val)
        est_aps.append(aps)
        est_bs.append(bs)
        est_sensitivity.append(sensitivity)
        est_specificity.append(specificity)
        est_ppv.append(ppv)

    # compute 95% CI for all classification performance
    confidence = 0.95
    cl_metrics = {'Accuracy': [round(np.mean(est_acc), 4), compute_ci(est_acc, confidence=confidence)],
                  'MCC': [round(np.mean(est_mcc), 4), compute_ci(est_mcc, confidence=confidence)],
                  'AUC': [round(np.mean(est_auc), 4), compute_ci(est_auc, confidence=confidence)],
                  'F1': [round(np.mean(est_f1), 4), compute_ci(est_f1, confidence=confidence)],
                  'BS': [round(np.mean(est_bs), 4), compute_ci(est_bs, confidence=confidence)],
                  }
    return cl_metrics

def compute_ci(data, confidence=0.95):
    """
    computer confidence interval for the given data
    """
    lower = np.percentile(data, (1 - confidence)/2 * 100)
    upper = np.percentile(data, (1 + confidence)/2 * 100)
    return [round(lower, 4), round(upper, 4)]

def main():
    """
    This program identifies probable negative examples and runs binary classifier on diagnosed positive
    and predicted negative examples to check the improvement in the classification performance.
    """
    iteration_count = 1

    # load IO filenames from the YAML file
    parser = argparse.ArgumentParser()
    parser.add_argument("-iofiles", default="io_data.yaml", help="provide yaml file containing io files")
    p_args = parser.parse_args()
    with open(p_args.iofiles, 'r') as fi:
        iodata = yaml.safe_load(fi)

    # STEP 1: load features, labels, covariates and person ids
    X_all, Y_all, rec_ids_all, covars_all = get_data_for_ml(features_file=iodata['output_files']['features_file'],
                                                            labels_file=iodata['output_files']['labels_file'],
                                                            persons_file=iodata['output_files']['persons_file'],
                                                            covars_file=iodata['output_files']['covars_file'])
    print(f"number of total records: {X_all.shape}, {len(Y_all)}")

    # STEP 2: select records with more than k features
    non_zero_count_per_row = np.diff(X_all.indptr)  # number of features per row
    rows_to_keep = non_zero_count_per_row >= iodata['nmf_vars']['min_feature_count']  # select rows with more than k features
    X_all, Y_all, rec_ids_all = X_all[rows_to_keep], Y_all[rows_to_keep], rec_ids_all[rows_to_keep]
    X_all_orig, Y_all_orig, rec_ids_all_orig = deepcopy(X_all), deepcopy(Y_all), deepcopy(rec_ids_all)  # save a copy
    print(f"number of records with >= {iodata['nmf_vars']['min_feature_count']} features: {X_all.shape}, {len(rec_ids_all)}, {len(Y_all)}")

    # STEP 3: select undiagnosed patients using given threshold
    for val in [10,50,100]: #range(20, 120, 20):
        r_threshold = val/100
        X_all, Y_all, rec_ids_all = deepcopy(X_all_orig), deepcopy(Y_all_orig), deepcopy(rec_ids_all_orig) # retrieve original
        X_all_updated, Y_all_updated = selected_undiagnosed_using_threshold(X_all, Y_all, rec_ids_all, risk_threshold=r_threshold,
                                                                            undiag_pred_file=iodata['output_files']['uncoded_patients_posterior_file'])
        print(f"data shape after selecting undiagnosed patients with risk <= {r_threshold}: {X_all_updated.shape}")

        # run classifier
        classification_metrics = run_binary_classifier(X_all_updated, Y_all_updated, rseed=1001, max_itereration=iteration_count)

        # print classification performance
        print(f"****** CLASSIFICATION RESULTS FOR THRESHOLD: {r_threshold} ******")
        for cl_metric, cl_val in classification_metrics.items():
            print(f"{cl_metric}: {cl_val}")

if __name__ == "__main__":
    main()

