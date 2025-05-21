import pickle
import numpy as np
import yaml
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
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
        # features
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


def get_prediction_data(diag_pred_file=None, undiag_pred_file=None):
    """
    load saved data from pickle files
    """
    print("\nfetch prediction data from the file")
    if diag_pred_file is None or undiag_pred_file is None:
        print("please provide file name for prediction data")
        exit(0)
    else:
        # diagnosed predictions
        with open(diag_pred_file, 'rb') as f:
            diag_pred_dict = pickle.load(f)

        # undiagnosed predictions
        with open(undiag_pred_file, 'rb') as f:
            undiag_pred_dict = pickle.load(f)

    return diag_pred_dict, undiag_pred_dict

def select_diagnosed_undiagnosed_person_features(diagnosed_preds, undiagnosed_preds, X_all, rec_ids_all, covars_all,
                                                 undiag_low_risk=True, diag_threshold=0.0, undiag_threshold=0.0):
    """
    This function selects all diagnosed patients with risk score >= diag_threshold,
    undiagnosed patients with risk >= undiag_threshold if undiag_low_risk is False, or <= undiag_threshold
    """
    print("\nselect individuals -- diagnosed and undiagnosed")
    diag_persons = np.asarray([pid for pid, pred in diagnosed_preds.items() if pred >= diag_threshold])
    if undiag_low_risk:
        undiag_persons = np.asarray([pid for pid, pred in undiagnosed_preds.items() if pred[0] <= undiag_threshold])
    else:
        undiag_persons = np.asarray([pid for pid, pred in undiagnosed_preds.items() if pred[0] >= undiag_threshold])

    # feature selection
    print("select frequent features for selected individuals -- diagnosed and undiagnosed")
    # make a copy of the original data
    X_all_orig, rec_ids_all_orig, covars_all_orig = deepcopy(X_all), deepcopy(rec_ids_all), deepcopy(covars_all)

    # diagnosed
    diag_pids_indx = np.isin(rec_ids_all, diag_persons).nonzero()[0]
    X_diagnosed = X_all[diag_pids_indx,:]
    nonzero_counts_diagnosed = np.array((X_diagnosed != 0).sum(axis=0)).flatten()  # Count of nonzero rows per feature
    sort_indx = np.argsort(nonzero_counts_diagnosed)[::-1] # sort indices by count in descending order
    covars_diagnosed = covars_all[sort_indx]

    # retrieve original copy
    X_all, rec_ids_all, covars_all = deepcopy(X_all_orig), deepcopy(rec_ids_all_orig), deepcopy(covars_all_orig)
    # undiagnosed
    undiag_pids_indx = np.isin(rec_ids_all, undiag_persons).nonzero()[0]
    X_undiagnosed = X_all[undiag_pids_indx,:]
    nonzero_counts_undiagnosed = np.array((X_undiagnosed != 0).sum(axis=0)).flatten()  # Count of nonzero rows per feature
    sort_indx = np.argsort(nonzero_counts_undiagnosed)[::-1] # sort indices by count in descending order
    covars_undiagnosed = covars_all[sort_indx]
    print(f"diagnosed matrix: {X_diagnosed.shape}, undiagnosed matrix: {X_undiagnosed.shape}")
    return covars_diagnosed, covars_undiagnosed


def compute_jaccard_similarity(covars1, covars2):
    """
    This function computes the jaccard_similarity between diagnosed and undiagnosed individuals' features
    """
    set1, set2 = set(covars1), set(covars2)
    intersection_len = len(set1 & set2)
    union_len = len(set1 | set2)
    if union_len == 0:
        return 0.0
    return intersection_len / union_len


def generate_line_plot(x, y_high, y_low, plotFile="jacc_plot.png"):
    """
    This function generates line plot using jaccard similarity and topK features
    """
    # Create the plot
    plt.plot(x, y_high, marker='v', linestyle='-', color='red', label='Diagnosed (High Risk) vs. Undiagnosed (High Risk)')
    plt.plot(x, y_low, marker='^', linestyle='-', color='blue', label='Diagnosed (High Risk) vs. Undiagnosed (Low Risk)')

    # Add labels and title
    plt.xlabel('Top K Covariates from Diagnosed and Undiagnosed Cohorts', fontsize=14)
    plt.ylabel('Jaccard Similarity', fontsize=14)
    plt.legend()

    # Set tick font sizes
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # Add grid with minor ticks
    plt.grid(which='major', linestyle='-', linewidth=0.5)
    plt.minorticks_on()
    plt.grid(which='minor', linestyle=':', linewidth=0.4)

    # Save the plot
    plt.tight_layout()
    plt.savefig(plotFile, dpi=300)

    # Show the plot
    plt.show()


def main():
    """
    This program selects diagnosed and undiagnosed individuals with elevated/low risk scores estimated by the model
    and compute feature similarity.
    """
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
    # select positive and unlabeled records
    print(f"number of total records: {X_all.shape}, {len(Y_all)}")

    # STEP 2: load predicted risk for diagnosed and undiagnosed patients
    diagnosed_preds, undiagnosed_preds = get_prediction_data(diag_pred_file=iodata['output_files']['coded_patients_posterior_file'],
                                                             undiag_pred_file=iodata['output_files']['uncoded_patients_posterior_file'])

    print(f"diagnosed count: {len(diagnosed_preds)}, undiagnosed count: {len(undiagnosed_preds)}")

    # STEP 3: select features of diagnosed (high risk) and undiagnosed patients (high risk)
    undiag_low_risk = False # select low risk undiagnosed? (e.g. when True, undiag_threshold=0.1, when False, undiag_threshold=0.9)
    undiag_threshold = 0.9  # select undiagnosed patients with risk >= undiag_threshold if undiag_low_risk is False, or <= undiag_threshold
    diag_threshold = 0.9    # select diagnosed patients with risk >= diag_threshold
    diag_covars, undiag_covars_high = select_diagnosed_undiagnosed_person_features(diagnosed_preds, undiagnosed_preds, X_all,
                                                                                   rec_ids_all, covars_all,
                                                                                   undiag_low_risk=undiag_low_risk,
                                                                                   diag_threshold=diag_threshold,
                                                                                   undiag_threshold=undiag_threshold)
    print(f"covars diagnosed: {len(diag_covars)}, covars undiagnosed high: {len(undiag_covars_high)}")

    # STEP 4: select features of diagnosed (high risk) and undiagnosed patients (low risk)
    undiag_low_risk = True # select low risk undiagnosed? (e.g. when True, undiag_threshold=0.1, when False, undiag_threshold=0.9)
    undiag_threshold = 0.1  # select undiagnosed patients with risk >= undiag_threshold if undiag_low_risk is False, or <= undiag_threshold
    diag_threshold = 0.9    # select diagnosed patients with risk >= diag_threshold
    diag_covars, undiag_covars_low = select_diagnosed_undiagnosed_person_features(diagnosed_preds, undiagnosed_preds, X_all,
                                                                                   rec_ids_all, covars_all,
                                                                                   undiag_low_risk=undiag_low_risk,
                                                                                   diag_threshold=diag_threshold,
                                                                                   undiag_threshold=undiag_threshold)
    print(f"covars diagnosed: {len(diag_covars)}, covars undiagnosed low: {len(undiag_covars_low)}")

    # STEP 5: select list of top k common features for diagnosed and undiagnosed patients
    xvals, yvals_high, yvals_low = [], [], []
    for k in range(10, 510, 10):
        # compute jaccard_similarity
        js_high = compute_jaccard_similarity(diag_covars[:k], undiag_covars_high[:k])
        js_low = compute_jaccard_similarity(diag_covars[:k], undiag_covars_low[:k])

        # append values
        xvals.append(k)
        yvals_high.append(js_high)
        yvals_low.append(js_low)

    # STEP 6: js vs k plot
    print(f"High vs High max jaccard: {max(yvals_high)}, min jaccard: {min(yvals_high)}")
    print(f"High vs Low max jaccard: {max(yvals_low)}, min jaccard: {min(yvals_low)}")
    generate_line_plot(xvals, yvals_high, yvals_low, plotFile=iodata['plot_files']['jacc_plot_file'])


if __name__ == "__main__":
    main()

