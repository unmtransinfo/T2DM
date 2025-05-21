import os.path
import pickle
import numpy as np
import yaml
import argparse
from sklearn.decomposition import NMF
from collections import Counter
import random
import gzip
from scipy.special import expit
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import percentileofscore
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def fetch_concept_id_name(ifile=None):
    """
    Read concept file and return a dictionary with concept_id as key and vocab_id + concept_name as value
    """
    print("\ncreating dictionary using concept id, vocab_id, and name")
    id_name_dict = {}

    # read the input file
    line_count = 0
    with gzip.open(ifile, 'rt') as fin:
        for line in fin:
            # track progress
            line_count += 1
            if line_count % 2500000 == 0:
                print("{0} concept records processed".format(line_count))

            # select values
            vals = line.strip().split('\t')
            if line_count == 1:
                pos_idx = {vv: ii for ii, vv in enumerate(vals)}
                cid_idx = pos_idx["concept_id"]
                cname_idx = pos_idx["concept_name"]
                vocab_idx = pos_idx["vocabulary_id"]
            else:
                cid, cname, cvocab = int(vals[cid_idx]), vals[cname_idx], vals[vocab_idx]
                id_name_dict[cid] = [cvocab, cname]

    print("{0} concept records processed".format(line_count))
    return id_name_dict


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


def select_train_test_set(X_pos, rec_ids_pos, rseed=1234, test_size=0.01, min_feature_count=1):
    """
    divide positive examples into train and test sets
    """
    print(f"select all rows with number of features >= {min_feature_count}")
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
    return X_pos[te_len:, :], X_pos[:te_len, :], rec_ids_pos[te_len:], rec_ids_pos[:te_len]

def select_rank_weighted_top_features_using_nmf(data, covars_list, n_clusters=1, min_feature_coeff=0, rseed=1, n_iterations=1):
    """
    Run NMF on diabetes cases and use H matrix to select top features
    """
    rank_weighted_feature_coeff = {}

    # determine rank-weighted coefficient for each feature
    for itr in range(n_iterations):
        print(f"iteration {itr+1}: run NMF on diabetes cases with {n_clusters} components")
        nmf = NMF(n_components=n_clusters, init='nndsvd', random_state=rseed*itr, max_iter=1000, l1_ratio=0)
        W = nmf.fit_transform(data)
        H = nmf.components_

        '''
        # Assign each patient to the cluster with the highest weight
        cluster_assignments = W.argmax(axis=1)  # Shape: (1.3M,)

        # Count patients per cluster
        cluster_counts = np.bincount(cluster_assignments)

        # Print results
        for cluster_idx in range(n_clusters):
            print(f"Cluster {cluster_idx}: {cluster_counts[cluster_idx]} patients ({cluster_counts[cluster_idx] / data.shape[0] * 100:.1f}%)")
        '''

        # determine top features for each component
        top_features = set()
        top_features_coeff = {}
        top_features_rank = {}
        for f in H:
            indices = np.argsort(f)[::-1]
            for r, j in enumerate(indices):
                top_features.add(covars_list[j])
                top_features_coeff[covars_list[j]] = top_features_coeff.get(covars_list[j], 0) + (f[j] / (r + 1))
                top_features_rank[covars_list[j]] = top_features_rank.get(covars_list[j], 0) + (1 / (r + 1))

        # compute rank-weighted coefficient
        print("compute rank-weighted coefficient")
        for f in top_features:
            vv = top_features_coeff[f] / top_features_rank[f]
            if vv >= min_feature_coeff:
                rank_weighted_feature_coeff[f] = rank_weighted_feature_coeff.get(f, 0) + vv

    # compute mean of rank weighted coefficient
    for feature, coeff in rank_weighted_feature_coeff.items():
        rank_weighted_feature_coeff[feature] = coeff/n_iterations
    return rank_weighted_feature_coeff


def save_top_nmf_features(all_nmf_features, concept_id_name, imp_covar_file=None, imp_covar_pkl_file=None,
                          imp_concept_id_name_file=None):
    """
    save features selected by NMF and their coefficients
    """
    # save concept_id_name for important features in a pickle file
    imp_feature_id_name_dict = {}
    with open(imp_concept_id_name_file, 'wb') as fi:
        for cov, coeff in all_nmf_features.items():
            imp_feature_id_name_dict[cov] = concept_id_name[cov]
        pickle.dump(imp_feature_id_name_dict, fi, protocol=5)

    # write features and coefficient to a pickle file
    with open(imp_covar_pkl_file, 'wb') as fp:
        pickle.dump(all_nmf_features, fp, protocol=5)

    # write features and coefficient to a TSV file
    with open(imp_covar_file, 'w') as fo:
        hdr = "Concept_id\tCoefficient\tConcept_name\n"
        fo.write(hdr)
        for vals in Counter.most_common(all_nmf_features):
            line = str(vals[0]) + "\t" + str(vals[1]) + "\t" + str(concept_id_name[vals[0]][1]) + "\n"
            fo.write(line)


def reshape_data_using_imp_features(X_tr, X_te, X_unlab, imp_covars_coeff, covars_all):
    """
    reshape train, test, and unlabeled example using important features
    """
    print("reshape data using important features")
    imp_covars = list(imp_covars_coeff.keys())
    idx = np.isin(covars_all, imp_covars).nonzero()[0]
    X_tr_reshaped = X_tr[:, idx]
    X_te_reshaped = X_te[:, idx]
    X_unlab_reshaped = X_unlab[:, idx]
    sel_covars = covars_all[idx]
    return X_tr_reshaped, X_te_reshaped, X_unlab_reshaped, sel_covars


def compute_feature_prevalence(csr_data):
    """
    Compute the prevalence of each feature in a CSR sparse matrix.
    Prevalence P(f) = (number of rows with nonzero value for f) / (total number of rows)
    """
    num_rows = csr_data.shape[0]
    nonzero_counts = np.array((csr_data != 0).sum(axis=0)).flatten()  # Count of nonzero rows per feature
    prevalence = nonzero_counts / num_rows
    return prevalence

def compute_KL_divergence(X_tr, X_unlab, sel_covars, diab_features, ofile=None, epsilon=1e-6):
    """
    Compute the KL divergence for each feature.
    """
    # feature prevalence in positive
    prev_p = compute_feature_prevalence(X_tr)

    # feature prevalence in unlabeled
    prev_u = compute_feature_prevalence(X_unlab)

    # Ensure values are within (0,1) range
    p = np.clip(prev_p, epsilon, 1 - epsilon)
    u = np.clip(prev_u, epsilon, 1 - epsilon)
    kl_div = p * np.log(p / u) + (1 - p) * np.log((1 - p) / (1 - u))

    # save selected important feature coefficient and divergence -- FOR WEB PORTAL
    covar_coeff_divergence = {}
    for j, covar in enumerate(sel_covars):
        covar_coeff_divergence[int(covar)] = [float(diab_features[covar]), float(kl_div[j])]
    with open(ofile, 'wb') as fo:
        pickle.dump(covar_coeff_divergence, fo, protocol=5)

    return kl_div

def select_test_records_with_prior_covars(rec_ids, imp_covars, features_file=None, persons_file=None, covars_file=None):
    """
    select test records with all covars prior to diagnosis date
    """
    # features
    with open(features_file, "rb") as fh:
        X_all = pickle.load(fh)

    # persons
    with open(persons_file, "rb") as fh:
        rec_ids_all = pickle.load(fh)

    # covars
    with open(covars_file, "rb") as fh:
        covars_all = pickle.load(fh)

    # find indices of important covariates in the list of all covariates
    covars_all_idx = {v: i for i, v in enumerate(covars_all)}
    imp_covars_idx = [covars_all_idx[v] for v in imp_covars]
    print(f"important covars count: {len(imp_covars_idx)}, {len(imp_covars)}")

    # find indices of all test records
    rec_indx = np.isin(rec_ids_all, rec_ids).nonzero()[0]
    print(f"test records count: {len(rec_indx)}, {len(rec_ids)}")

    # select important features for all test records
    X_test = X_all[rec_indx,:][:,imp_covars_idx]
    print(f"shape of test records: {X_test.shape}")

    return X_test, rec_ids_all[rec_indx]

def compute_posterior(patient_features, feature_coeffs, feature_likelihoods, rec_ids, bias_term=0):
    """
    Compute posterior probability of T2D for a patient based on important features.
    """
    # compute element-wise product of feature coefficients and likelihoods
    weights = feature_coeffs * feature_likelihoods

    # compute score as dot product of sparse feature matrix and weights
    scores = patient_features.dot(weights) + bias_term

    # apply transformation to get posterior probability
    probs = np.arctan(scores) / (np.pi / 2)

    return dict(zip(rec_ids, probs))


def compute_percentile_scores(diagnosed_scores, unalab_pred_dict):
    """
    compute percentile for predicted scores of undiagnosed patients
    """
    print(f"compute percentile for {len(unalab_pred_dict)} records")
    diagnosed_scores_sorted = np.sort(diagnosed_scores)
    person_ids, risk_scores = zip(*unalab_pred_dict.items())

    # find positions for undiagnosed scores in diagnosed scores
    positions = np.searchsorted(diagnosed_scores_sorted, risk_scores, side='right')

    # convert positions to percentile
    percentiles = (positions / len(diagnosed_scores_sorted)) * 100

    # create a dictionary
    pid_percentile_score_dict = {}
    for j, p in enumerate(person_ids):
        pid_percentile_score_dict[p] = [risk_scores[j], percentiles[j]]

        # check progress
        #if (j+1)%5000000 == 0:
        #    print(f"dictionary created for {j+1} records")
    #print(f"dictionary created for {j + 1} records")

    return pid_percentile_score_dict


def main():
    """
    - This program uses NMF to determine the list of important features and their coefficients.
    - computes likelihood of each important feature.
    - using coefficient and likelihood of each feature to compute the probability of T2D for uncoded patients.
    """
    # load IO filenames from the YAML file
    parser = argparse.ArgumentParser()
    parser.add_argument("-iofiles", default="io_data.yaml", help="provide yaml file containing io files")
    p_args = parser.parse_args()
    with open(p_args.iofiles, 'r') as fi:
        iodata = yaml.safe_load(fi)

    # STEP 1: read concept file and create a dictionary with concept_id as key and vocab_id + concept_name as values
    concept_id_name_dict = {}
    if not os.path.exists(iodata['output_files']['concept_id_name_file']):
        concept_id_name_dict = fetch_concept_id_name(ifile=iodata['input_files']['concepts_file'])
    else:
        with open(iodata['output_files']['concept_id_name_file'], 'rb') as f:
            concept_id_name_dict = pickle.load(f)
    print("number of concepts: ", len(concept_id_name_dict))

    # STEP 2: get data for ml
    X_all, Y_all, rec_ids_all, covars_all = get_data_for_ml(features_file=iodata['output_files']['features_file'],
                                                            labels_file=iodata['output_files']['labels_file'],
                                                            persons_file=iodata['output_files']['persons_file'],
                                                            covars_file=iodata['output_files']['covars_file'])
    # select positive and unlabeled records
    print(f"number of total records: {X_all.shape}, {len(Y_all)}")
    X_pos, X_unlab = X_all[Y_all == 1], X_all[Y_all == 0]
    rec_ids_pos, rec_ids_unlab = rec_ids_all[Y_all == 1], rec_ids_all[Y_all == 0]
    nonzero_counts_per_row = X_pos.getnnz(axis=1)
    print(f"number of positive records: {np.sum(Y_all)}, {X_pos.shape}, {len(rec_ids_pos)}, unlabeled examples: {X_unlab.shape}")
    print(f"average number of non-zero elements in X_pos: {int(np.mean(nonzero_counts_per_row))}")

    # divide positive examples into train and test sets - select records with >=3 features
    X_tr, X_te, rec_ids_tr, rec_ids_te = select_train_test_set(X_pos, rec_ids_pos, rseed=iodata['nmf_vars']['rseed_val'],
                                                               test_size=iodata['nmf_vars']['test_size'],
                                                               min_feature_count=iodata['nmf_vars']['min_feature_count'])
    print(f"number of train records: {X_tr.shape}, {len(rec_ids_tr)} and test records records: {X_te.shape}, {len(rec_ids_te)}")

    # STEP 3: use NMF on positive examples to determine co-occurring conditions and drug prescriptions
    diab_features_temp = {}
    if not os.path.exists(iodata['output_files']['imp_features_pkl_file']):
        diab_features_temp = select_rank_weighted_top_features_using_nmf(X_tr, covars_all, n_clusters=iodata['nmf_vars']['n_clusters'],
                                                                         min_feature_coeff=iodata['nmf_vars']['min_feature_coeff'],
                                                                         rseed=iodata['nmf_vars']['rseed_val'],
                                                                         n_iterations=iodata['nmf_vars']['n_iterations'])
        save_top_nmf_features(diab_features_temp, concept_id_name_dict, imp_covar_file=iodata['output_files']['imp_features_file'],
                              imp_covar_pkl_file=iodata['output_files']['imp_features_pkl_file'],
                              imp_concept_id_name_file=iodata['output_files']['concept_id_name_file'])
    else:
        with open(iodata['output_files']['imp_features_pkl_file'], 'rb') as f:
            diab_features_temp = pickle.load(f)
    print(f"number of top features selected using NMF with {iodata['nmf_vars']['n_clusters']} clusters: {len(diab_features_temp)}")

    # select features with coefficient > mean value
    mean_rwc = np.mean([v for k, v in diab_features_temp.items() if v > 0])
    diab_features = {k:v for k, v in diab_features_temp.items() if v >= mean_rwc}
    print(f"number of important features with RWC>{mean_rwc}: {len(diab_features)}")

    # STEP 4: compute prevalence of each selected important features
    # reshape data using important features
    X_tr, X_te, X_unlab, sel_covars = reshape_data_using_imp_features(X_tr, X_te, X_unlab, diab_features, covars_all)
    sel_covars_coeff = np.asarray([diab_features[v] for v in sel_covars])
    print(f"reshaped train:{X_tr.shape}, test: {X_te.shape}, unlabeled:{X_unlab.shape}, covariates:{len(sel_covars)}")
    exit()

    # compute Kullback–Leibler Divergence (KL Divergence) for each feature
    f_divergence = compute_KL_divergence(X_tr, X_unlab, sel_covars, diab_features,
                                         ofile=iodata['output_files']['imp_covar_coeff_likelihood'], epsilon=1e-20)

    # STEP 5: estimate the prediction for records
    # compute posterior for train patients
    print("\ncompute posterior probability for train records")
    X_tr.data[:] = 1  # set non-zero value to 1
    train_pred_dict = compute_posterior(X_tr, sel_covars_coeff, f_divergence, rec_ids_tr, bias_term=0)
    posterior_tr = np.asarray(list(train_pred_dict.values()))
    print(f"min train posterior: {np.min(posterior_tr)}, max train posterior: {np.max(posterior_tr)}")
    print(f"mean train posterior: {np.mean(posterior_tr)}, median train posterior: {np.median(posterior_tr)}")
    # save train posterior values
    with open(iodata['output_files']['coded_patients_posterior_file'], 'wb') as fp:
        pickle.dump(train_pred_dict, fp, protocol=5)

    # compute posterior for unlabeled patients
    print("\ncompute posterior probability for unlabeled records")
    X_unlab.data[:] = 1  # set non-zero value to 1
    unalab_pred_dict = compute_posterior(X_unlab, sel_covars_coeff, f_divergence, rec_ids_unlab, bias_term=0)
    posterior_un = np.asarray(list(unalab_pred_dict.values()))
    print(f"min unlabeled posterior: {np.min(posterior_un)}, max unlabeled posterior: {np.max(posterior_un)}")
    print(f"mean unlabeled posterior: {np.mean(posterior_un)}, median unlabeled posterior: {np.median(posterior_un)}")
    # compute percentile for each predicted risk score
    percentile_scores = compute_percentile_scores(posterior_tr, unalab_pred_dict)
    # save unlabeled posterior values
    with open(iodata['output_files']['uncoded_patients_posterior_file'], 'wb') as fp:
        pickle.dump(percentile_scores, fp, protocol=5)

    # compute posterior for test patients
    print("\ncompute posterior probability for test records")
    X_te.data[:] = 1  # set non-zero value to 1
    test_pred_dict = compute_posterior(X_te, sel_covars_coeff, f_divergence, rec_ids_te, bias_term=0)
    posterior_te = np.asarray(list(test_pred_dict.values()))
    print(f"min test_all posterior: {np.min(posterior_te)}, max test_all posterior: {np.max(posterior_te)}")
    print(f"mean test_all posterior: {np.mean(posterior_te)}, median test_all posterior: {np.median(posterior_te)}")
    # compute percentile for each predicted risk score
    percentile_scores = compute_percentile_scores(posterior_tr, test_pred_dict)
    # save test posteriors
    with open(iodata['output_files']['test_patients_posterior_file'], 'wb') as fp:
        pickle.dump(percentile_scores, fp, protocol=5)


if __name__ == "__main__":
    main()

