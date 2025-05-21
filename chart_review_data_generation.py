import pickle
import yaml
import argparse
import numpy as np
import random
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


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

def unlab_records_for_chart_review(undiagnosed_preds):
    """
    select one records from each of 100 risk score bins
    """
    bin_pid_dict = {v:[] for v in range(0,101,1)}
    # bin_count_dict = {v:0 for v in range(0,101,1)}

    # shuffle dictionary
    itemss = list(undiagnosed_preds.items())
    random.shuffle(itemss)
    shuffled_undiagnosed_preds = dict(itemss)

    # select undiagnosed pids
    for pid, pred in shuffled_undiagnosed_preds.items():
        all_bins_filled = all(len(v) > 0 for v in bin_pid_dict.values())    # check if all bins are filled
        # bin_count_dict[round(pred[1])]+=1
        if not all_bins_filled:
            for k, v in bin_pid_dict.items():
                if len(v) == 0:
                    bin_pid_dict[round(pred[1])].append(pid)
                    # print(round(pred[1]), pid)
                    break
        else:
            print("all bins are filled, exiting loop")
            break

    # print(bin_count_dict)
    selected_persons = [v[0] for _, v in bin_pid_dict.items()]
    return np.asarray(selected_persons)


def generate_histogram_data(diagnosed_preds):
    """
    generate data for histogram
    """
    bin_pid_count_dict = {v / 10: 0 for v in range(1, 11, 1)}
    for pid, pred in diagnosed_preds.items():
        for k, v in bin_pid_count_dict.items():
            if k - 0.1 <= pred < k:
                bin_pid_count_dict[k]+=1

    for k,v in bin_pid_count_dict.items():
        print(str(round(k-0.1,1))+"-"+str(round(k,1)) + "\t" + str(v))

def generate_chart_review_data(X, unlab_recs, covars_imp, undiagnosed_preds, concept_id_name_dict, ofile=None):
    """
    create data for chart review
    """
    with open(ofile, "w") as fo:
        for j, pid in enumerate(unlab_recs):
            # print(f"selected pid {pid} for chart review from percentile {round(undiagnosed_preds[pid][1])}")
            nonzero_idx = X[j].nonzero()[1]
            for covar in covars_imp[nonzero_idx]:
                line = str(pid) + "\t" + str(round(undiagnosed_preds[pid][1])) + "\t" + str(concept_id_name_dict[covar][1]) + "\n"
                fo.write(line)

def main():
    """
    - This program randomly selects 100 patients, 1 from each percentile bin, for chart review.
    """
    # load IO filenames from the YAML file
    parser = argparse.ArgumentParser()
    parser.add_argument("-iofiles", default="io_data.yaml", help="provide yaml file containing io files")
    p_args = parser.parse_args()
    with open(p_args.iofiles, 'r') as fi:
        iodata = yaml.safe_load(fi)

    # STEP 1: load concept_id and concept_name dictionary
    with open(iodata['output_files']['concept_id_name_file'], 'rb') as f:
        concept_id_name_dict = pickle.load(f)
    print("number of covariates selected using NMF: ", len(concept_id_name_dict))

    # STEP 2: get data for ml
    X_all, Y_all, rec_ids_all, covars_all = get_data_for_ml(features_file=iodata['output_files']['features_file'],
                                                            labels_file=iodata['output_files']['labels_file'],
                                                            persons_file=iodata['output_files']['persons_file'],
                                                            covars_file=iodata['output_files']['covars_file'])
    # select positive and unlabeled records
    print(f"number of total records: {X_all.shape}, {len(Y_all)}")

    # STEP 3: load predicted risk for diagnosed and undiagnosed patients
    diagnosed_preds, undiagnosed_preds = get_prediction_data(diag_pred_file=iodata['output_files']['coded_patients_posterior_file'],
                                                             undiag_pred_file=iodata['output_files']['uncoded_patients_posterior_file'])
    print(f"diagnosed count: {len(diagnosed_preds)}, undiagnosed count: {len(undiagnosed_preds)}")
    
    # check distribution of diagnosed patients
    # generate_histogram_data(diagnosed_preds)

    # STEP 4: select all important covariates
    with open(iodata['output_files']['imp_features_pkl_file'], 'rb') as f:
        diab_features_temp = pickle.load(f)
    # select features with coefficient > mean value
    mean_rwc = np.mean([v for k, v in diab_features_temp.items() if v > 0])
    imp_covars = [k for k, v in diab_features_temp.items() if v >= mean_rwc]
    c_idx = np.isin(covars_all, imp_covars).nonzero()[0]
    print(f"number of important features with RWC>{mean_rwc}: {len(imp_covars)}, {len(c_idx)}")

    # STEP 5: for each probability bin, select one unlabeled record
    selected_unlab_recs = unlab_records_for_chart_review(undiagnosed_preds)
    print("number of patients selected for chart review: ", len(selected_unlab_recs))

    # create data for chart review
    r_idx = np.isin(rec_ids_all, selected_unlab_recs).nonzero()[0]
    selected_X = X_all[r_idx,:][:, c_idx]   # select feature matrix using rows of patients and columns of imp features
    selected_unlab_recs = rec_ids_all[r_idx]  # selected patient id
    imp_covars = covars_all[c_idx]  # selected features
    print("create file for k selected patients: ", selected_X.shape, len(selected_unlab_recs))
    generate_chart_review_data(selected_X, selected_unlab_recs, imp_covars, undiagnosed_preds, concept_id_name_dict,
                               ofile=iodata['output_files']['chart_review_file'])


if __name__ == "__main__":
    main()

