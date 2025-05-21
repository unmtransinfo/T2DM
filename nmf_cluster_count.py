import pickle
import yaml
import argparse
import random
from sklearn.decomposition import NMF
import numpy as np
import matplotlib.pyplot as plt



def get_data_for_ml(features_file=None, labels_file=None, persons_file=None, covars_file=None):
    """
    Fetch saved data for ML method
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


def select_train_test_set(X_pos, rec_ids_pos, rseed=7, test_size=0.0, min_feature_count=1):
    """
    divide positive examples into train and test sets
    """
    # X_pos: drop all rows that has less than min_feature_count features
    non_zero_count_per_row = np.diff(X_pos.indptr)
    rows_to_keep = non_zero_count_per_row >= min_feature_count
    X_pos = X_pos[rows_to_keep]
    rec_ids_pos = rec_ids_pos[rows_to_keep]

    # divide positive examples into train and test sets
    random.seed(rseed)
    indx = random.sample(range(X_pos.shape[0]), X_pos.shape[0]) # shuffle indices
    X_pos, rec_ids_pos = X_pos[indx], rec_ids_pos[indx] # shuffle positive examples
    te_len = int(len(indx) * test_size)
    return X_pos[te_len:, :], rec_ids_pos[:te_len],



def main():
    """
    - This program estimates the number of clusters for NMF
    """
    # load IO filenames from the YAML file
    parser = argparse.ArgumentParser()
    parser.add_argument("-iofiles", default="io_data.yaml", help="provide yaml file containing io files")
    p_args = parser.parse_args()
    with open(p_args.iofiles, 'r') as fi:
        iodata = yaml.safe_load(fi)

    # STEP 1: get data for ml
    X_all, Y_all, rec_ids_all, covars_all = get_data_for_ml(features_file=iodata['output_files']['features_file'],
                                                            labels_file=iodata['output_files']['labels_file'],
                                                            persons_file=iodata['output_files']['persons_file'],
                                                            covars_file=iodata['output_files']['covars_file'])
    # select positive and unlabeled records
    print(f"number of total records: {X_all.shape}, {len(Y_all)}")
    X_pos = X_all[Y_all == 1]
    X_unlab = X_all[Y_all == 0]
    rec_ids_pos = rec_ids_all[Y_all == 1]
    nonzero_counts_per_row = X_pos.getnnz(axis=1)
    print(f"number of positive records: {np.sum(Y_all)}, {X_pos.shape}, {len(rec_ids_pos)}, unlabeled examples: {X_unlab.shape}")
    print(f"average number of non-zero elements in X_pos: {int(np.mean(nonzero_counts_per_row))}")

    # divide positive examples into train and test sets - select records with >=5 features
    X_tr, rec_ids_te = select_train_test_set(X_pos, rec_ids_pos, rseed=iodata['nmf_vars']['rseed_val'],
                                             test_size=iodata['nmf_vars']['test_size'],
                                             min_feature_count=iodata['nmf_vars']['min_feature_count'])
    print(f"number of train records: {X_tr.shape} and test records: {len(rec_ids_te)}")

    # determine number of clusters
    errors = []
    cluster_vals = range(2, 16)
    for k in cluster_vals:
        print(f"processing for cluster count: {k}, {errors}")
        model = NMF(n_components=k, init='nndsvd', random_state=1001, max_iter=1000, l1_ratio=0)
        W = model.fit_transform(X_tr)
        errors.append(model.reconstruction_err_)

    err_diff = np.diff(errors)*-1
    print(err_diff)
    plt.plot(cluster_vals, errors, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Reconstruction Error')
    plt.title('Elbow Method for NMF')
    plt.show()

if __name__ == "__main__":
    main()

