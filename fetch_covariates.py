import numpy as np
import pandas as pd
from datetime import datetime
import pickle
import os
import yaml
import argparse
import gzip
from scipy.sparse import csr_matrix, hstack, lil_matrix


def fetch_concept_id_name(ifile=None):
    """
    Read concept file and return a dictionary with concept_id as key and concept_name + vocab_id as value
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


def fetch_observation_dates(ifile=None, ofile=None):
    """
    for each patient, select observation start and end date
    """
    print("\nselect observation start and end dates for all patients")
    person_first_last_obs_dict = {}

    # read large file
    line_count = 0
    with gzip.open(ifile, 'rt') as fin:
        for line in fin:
            # track progress
            line_count += 1
            if line_count % 5000000 == 0:
                print("{0} records processed to determine observation dates".format(line_count))

            # select values
            vals = line.strip().split('\t')
            if line_count == 1:
                pos_idx = {vv: ii for ii, vv in enumerate(vals)}
                pid_idx = pos_idx["person_id"]
                s_obs_date_idx = pos_idx["observation_period_start_date"]
                e_obs_date_idx = pos_idx["observation_period_end_date"]
            else:
                # get start and end observation dates for patient
                pid, p_obs_start_date, p_obs_end_date = int(vals[pid_idx]), vals[s_obs_date_idx], vals[e_obs_date_idx]
                person_first_last_obs_dict[pid] = [p_obs_start_date, p_obs_end_date]

    print("{0} records processed to determine observation dates".format(line_count))

    # save dictionary
    with open(ofile, 'wb') as f:
        pickle.dump(person_first_last_obs_dict, f, protocol=5)

    return person_first_last_obs_dict


def get_person_details(p_obs_dates_dict, ifile=None, ofile=None):
    """
    fetch all selected persons' data (age, sex, state) from the input file
    """
    print("\nfetch person gender, age and location from input file")
    person_data_dict = {}
    loc_id_state = {'1': 'AK', '2': 'AL', '3': 'AR', '4': 'AZ', '5': 'CA', '6': 'CO', '7': 'CT', '8': 'DC',
                    '9': 'DE', '10': 'FL', '11': 'GA', '12': 'HI', '13': 'IA', '14': 'ID', '15': 'IL', '16': 'IN',
                    '17': 'KS', '18': 'KY', '19': 'LA', '20': 'MA', '21': 'MD', '22': 'ME', '23': 'MI', '24': 'MN',
                    '25': 'MO', '26': 'MS', '27': 'MT', '28': 'NC', '29': 'ND', '30': 'NE', '31': 'NH', '32': 'NJ',
                    '33': 'NM', '34': 'NV', '35': 'NY', '36': 'OH', '37': 'OK', '38': 'OR', '39': 'PA', '40': 'PR',
                    '41': 'RI', '42': 'SC', '43': 'SD', '44': 'TN', '45': 'TX', '46': 'UN', '47': 'UN', '48': 'UN',
                    '49': 'UN', '50': 'UN', '51': 'UN', '52': 'UN', '53': 'UT', '54': 'VA', '55': 'VT', '56': 'WA',
                    '57': 'WI', '58': 'WV', '59': 'WY', '60': 'UN'} # copied from CARC machine CCAE database (4/18/2025)

    # read the file
    line_count = 0
    with gzip.open(ifile, 'rt') as fin:
        for line in fin:
            # track progress
            line_count += 1
            if line_count % 5000000 == 0:
                print("{0} persons details were fetched".format(line_count))

            # select values
            vals = line.strip().split('\t')
            if line_count == 1:
                pos_idx = {vv: ii for ii, vv in enumerate(vals)}
                pid_idx = pos_idx["person_id"]
                gender_idx = pos_idx["gender_concept_id"]
                yob_idx = pos_idx["year_of_birth"]
                location_id = pos_idx["location_id"]
            else:
                pid = int(vals[pid_idx])
                if pid not in p_obs_dates_dict:
                    continue
                else:
                    obs_date = datetime.strptime(p_obs_dates_dict[pid][0], "%Y-%m-%d")
                    p_age = obs_date.year - int(vals[yob_idx])  # age
                    p_gender = 1 if vals[gender_idx] == '8507' else 0

                    # some persons do not have state information - it will handle those data
                    if len(vals) < 4:
                        p_state = 'UN'
                    else:
                        p_state = loc_id_state[vals[location_id]]

                    # update dictionary
                    person_data_dict[pid] = [p_gender, p_age, p_state]

    print("{0} persons details were fetched".format(line_count))

    # save dictionary
    with open(ofile, 'wb') as f:
        pickle.dump(person_data_dict, f, protocol=5)

    return person_data_dict


def select_diabetes_codes(ifile1=None, ifile2=None):
    """
    select all type 1 and type 2 diabetes ICD10 codes
    """
    # type 1 and 2 codes
    typ1_diab_df = pd.read_csv(ifile1, sep="\t", header=0, compression='gzip')
    typ2_diab_df = pd.read_csv(ifile2, sep="\t", header=0, compression='gzip')
    type1_diab = typ1_diab_df['concept_id'].to_numpy(dtype=int)
    type2_diab = typ2_diab_df['concept_id'].to_numpy(dtype=int)
    return set(type1_diab), set(type2_diab)

def date_to_object(p_date_dict, n_elem=1):
    """
    convert date from YYYY-MM-DD format to datetime object
    """
    dateobj_dict = {}
    for kk, vv in p_date_dict.items():
        if n_elem == 1: # first diabetes date
            dateobj_dict[kk] = datetime.strptime(vv, "%Y-%m-%d")
        elif n_elem == 2: # observation dates
            dateobj_dict[kk] = [datetime.strptime(vv[0], "%Y-%m-%d"), datetime.strptime(vv[1], "%Y-%m-%d")]
    return dateobj_dict


def fetch_first_diab_coding_date(typ1_diab_codes, typ2_diab_codes, pp_obs_date_dict, icd9_diab_codes=None, ifile=None, ofile1=None, ofile2=None):
    """
    read condition file and determine the very first diabetes coding date for diabetes persons
    """
    print("\ncreating a dictionary using person_id and first diabetes date")
    p_diab_date_dict = {}
    excluded_pid = set()

    # convert observation dates to object
    p_obs_date_dict = date_to_object(pp_obs_date_dict, n_elem=2)

    # read large file
    line_count = 0
    with gzip.open(ifile, 'rt') as fin:
        for line in fin:
            # track progress
            line_count += 1
            if line_count % 25000000 == 0:
                print("{0} records processed to determine first diabetes coding date".format(line_count))

            # select values
            vals = line.strip().split('\t')
            if line_count == 1:
                pos_idx = {vv: ii for ii, vv in enumerate(vals)}
                pid_idx = pos_idx["person_id"]
                cond_idx = pos_idx["condition_concept_id"]
                cond_src_idx = pos_idx["condition_source_concept_id"]
                cond_date_idx = pos_idx["condition_start_date"]
            else:
                if vals[cond_idx] == '0':   # skip bad codes
                    continue

                # get data from the line
                pid, p_cond, p_cond_date = int(vals[pid_idx]), int(vals[cond_src_idx]), vals[cond_date_idx]

                # exclude all type 1 patients, persons coded with ICD9
                if p_cond in typ1_diab_codes or p_cond in icd9_diab_codes or pid in excluded_pid:
                    excluded_pid.add(pid)
                else:
                    # skip person with no observation or not type 2 diabetes code
                    if pid in p_obs_date_dict and p_cond in typ2_diab_codes:
                        obs_start_date = p_obs_date_dict[pid][0]
                        obs_end_date = p_obs_date_dict[pid][1]
                        cond_date = datetime.strptime(p_cond_date, "%Y-%m-%d")

                        # check if condition occurred within observation period
                        if obs_start_date <= cond_date <= obs_end_date:
                            if pid not in p_diab_date_dict:
                                p_diab_date_dict[pid] = p_cond_date
                            else:
                                prev_cond_date = datetime.strptime(p_diab_date_dict[pid], "%Y-%m-%d")
                                if cond_date < prev_cond_date:
                                    p_diab_date_dict[pid] = p_cond_date
    print("{0} records processed to determine first diabetes coding date".format(line_count))

    # save dictionary
    with open(ofile1, 'wb') as f:
        pickle.dump(p_diab_date_dict, f, protocol=5)

    # save set
    with open(ofile2, 'wb') as f:
        pickle.dump(excluded_pid, f, protocol=5)
    return p_diab_date_dict, excluded_pid


def get_person_conditions(typ2_diab_codes, pp_diab_date_dict, pp_obs_date_dict, concept_id_vocab_dict, excluded_pids, ifile=None, ofile=None, covar_selection='all'):
    """
    fetch person conditions from the input file.
    """
    print(f"\nselect {covar_selection} condition covariates for persons")
    person_conditions_dict = {}

    print("covert dates to datetime object")
    # convert first diabetes date to object
    p_diab_date_dict = date_to_object(pp_diab_date_dict, n_elem=1)

    # convert observation dates to object
    p_obs_date_dict = date_to_object(pp_obs_date_dict, n_elem=2)

    print("read the file and process records")
    # read the file
    line_count = 0
    with gzip.open(ifile, 'rt') as fin:
        for line in fin:
            # track progress
            line_count += 1
            if line_count % 25000000 == 0:
                print("fetched conditions from {0} records".format(line_count))

            # select values
            vals = line.strip().split('\t')
            if line_count == 1:
                pos_idx = {vv: ii for ii, vv in enumerate(vals)}
                pid_idx = pos_idx["person_id"]
                cond_idx = pos_idx["condition_concept_id"]
                cond_src_idx = pos_idx["condition_source_concept_id"]
                cond_date_idx = pos_idx["condition_start_date"]
            else:
                if vals[cond_idx] == '0':   # skip bad codes
                    continue

                # get data from the line
                pid, p_src_cond, p_cond_date = int(vals[pid_idx]), int(vals[cond_src_idx]), vals[cond_date_idx]

                # skip all excluded persons and do not include type 2 code in the list of covariates
                if p_src_cond in typ2_diab_codes or pid in excluded_pids:
                    continue

                # select only persons who were observed
                if pid in p_obs_date_dict:
                    obs_start_date = p_obs_date_dict[pid][0]
                    obs_end_date = p_obs_date_dict[pid][1]
                    cond_date = datetime.strptime(p_cond_date, "%Y-%m-%d")

                    # condition should occur between observation period and should be ICD10CM
                    if obs_start_date <= cond_date <= obs_end_date and concept_id_vocab_dict[p_src_cond][0] == 'ICD10CM':
                        # initialize the dictionary for the pid and condition code
                        person_conditions_dict.setdefault(pid, {}).setdefault(p_src_cond, 0)

                        # populate the dictionary
                        if covar_selection == 'all':
                            person_conditions_dict[pid][p_src_cond] += 1
                        else:
                            if pid in p_diab_date_dict:  # cases - condition should be before diagnose date
                                if cond_date < p_diab_date_dict[pid]:
                                    person_conditions_dict[pid][p_src_cond] += 1
                            else:   # controls
                                person_conditions_dict[pid][p_src_cond] += 1
    print("fetched conditions from {0} records".format(line_count))

    # save dictionary
    with open(ofile, 'wb') as f:
        pickle.dump(person_conditions_dict, f, protocol=5)

    return person_conditions_dict

def get_person_drugs(pp_diab_date_dict, pp_obs_date_dict, excluded_pids, ifile=None, ofile=None, covar_selection='all'):
    """
    fetch drugs for each person from the input file
    """
    print(f"\nselect {covar_selection} drug covariates for persons")
    person_drugs_dict = {}

    print("covert dates to datetime object")
    # convert first diabetes date to object
    p_diab_date_dict = date_to_object(pp_diab_date_dict, n_elem=1)

    # convert observation dates to object
    p_obs_date_dict = date_to_object(pp_obs_date_dict, n_elem=2)

    print("read the file and process records")
    # read the file
    line_count = 0
    with gzip.open(ifile, 'rt') as fin:
        for line in fin:
            # track progress
            line_count += 1
            if line_count % 25000000 == 0:
                print("fetched drugs from {0} records".format(line_count))

            # select values
            vals = line.strip().split('\t')
            if line_count == 1:
                pos_idx = {vv: ii for ii, vv in enumerate(vals)}
                pid_idx = pos_idx["person_id"]
                drug_idx = pos_idx["drug_concept_id"]
                drug_date_idx = pos_idx["drug_era_start_date"]
                drug_exposure_idx = pos_idx["drug_exposure_count"]
            else:
                if vals[drug_idx] == '0':   # skip bad codes
                    continue

                # get data from line
                pid, drug_concept_id = int(vals[pid_idx]), int(vals[drug_idx])
                drug_start_date, drug_exposure_count = vals[drug_date_idx], int(vals[drug_exposure_idx])

                # skip all excluded persons
                if pid in excluded_pids:
                    continue

                # select only observed persons
                if pid in p_obs_date_dict:
                    obs_start_date = p_obs_date_dict[pid][0]
                    obs_end_date = p_obs_date_dict[pid][1]
                    d_date = datetime.strptime(drug_start_date, "%Y-%m-%d")

                    # drug date should be within observation period
                    if obs_start_date <= d_date <= obs_end_date:
                        # initialize the dictionary for the pid and drug concept code
                        person_drugs_dict.setdefault(pid, {}).setdefault(drug_concept_id, 0)

                        # populate the dictionary
                        if covar_selection == 'all':
                            person_drugs_dict[pid][drug_concept_id] += 1 #drug_exposure_count
                        else:
                            if pid in p_diab_date_dict: # cases
                                if d_date < p_diab_date_dict[pid]:
                                    person_drugs_dict[pid][drug_concept_id] += 1 # drug_exposure_count
                            else:   # controls
                                person_drugs_dict[pid][drug_concept_id] += 1 # drug_exposure_count

    print("fetched drugs from {0} records".format(line_count))

    # save dictionary
    with open(ofile, 'wb') as f:
        pickle.dump(person_drugs_dict, f, protocol=5)

    return person_drugs_dict


def get_person_procedures(pp_diab_date_dict, pp_obs_date_dict, excluded_pids, ifile=None, ofile=None):
    """
    fetch procedures for each person from the input file
    """
    print("\nselect procedure covariates for persons")
    person_procs_dict = {}

    print("convert dates to datetime object")
    # convert first diabetes date to object
    p_diab_date_dict = date_to_object(pp_diab_date_dict, n_elem=1)

    # convert observation dates to object
    p_obs_date_dict = date_to_object(pp_obs_date_dict, n_elem=2)

    print("read the file and process records")
    # read the file
    line_count = 0
    with gzip.open(ifile, 'rt') as fin:
        for line in fin:
            # track progress
            line_count += 1
            if line_count % 25000000 == 0:
                print("fetched procedures from {0} records".format(line_count))

            # select values
            vals = line.strip().split('\t')
            if line_count == 1:
                pos_idx = {vv: ii for ii, vv in enumerate(vals)}
                pid_idx = pos_idx["person_id"]
                proc_idx = pos_idx["procedure_concept_id"]
                proc_date_idx = pos_idx["procedure_date"]
            else:
                if vals[proc_idx] == '0':   # skip bad codes
                    continue

                # get data from line
                pid, proc_concept_id, proc_date = int(vals[pid_idx]), int(vals[proc_idx]), vals[proc_date_idx]

                # skip all excluded persons
                if pid in excluded_pids:
                    continue

                # select only observed persons
                if pid in p_obs_date_dict:
                    obs_start_date = p_obs_date_dict[pid][0]
                    obs_end_date = p_obs_date_dict[pid][1]
                    d_date = datetime.strptime(proc_date, "%Y-%m-%d")

                    # proc date should be within observation period
                    if obs_start_date < d_date < obs_end_date:
                        # initialize the dictionary for the pid and drug concept code
                        if pid not in person_procs_dict:
                            person_procs_dict.setdefault(pid, {})
                        if proc_concept_id not in person_procs_dict[pid]:
                            person_procs_dict[pid].setdefault(proc_concept_id, 0)

                        # populate the dictionary
                        if pid in p_diab_date_dict: # cases
                            diab_date = p_diab_date_dict[pid]
                            if d_date < diab_date:
                                person_procs_dict[pid][proc_concept_id] += 1
                        else:   # controls
                            person_procs_dict[pid][proc_concept_id] += 1
    print("fetched procedures from {0} records".format(line_count))

    # save dictionary
    with open(ofile, 'wb') as f:
        pickle.dump(person_procs_dict, f, protocol=5)

    return person_procs_dict


def generate_feature_and_label(p_diab_date_dict, excluded_pid, person_conditions=None, person_drugs=None, person_procs=None):
    """
    generate features in CSR format for ML
    """
    print("\ngenerate data for ML models")
    if person_conditions is None:
        person_conditions = {}
    if person_drugs is None:
        person_drugs = {}
    if person_procs is None:
        person_procs = {}

    # unique persons
    all_persons = set(list(person_conditions.keys()))
    all_persons.update(list(person_drugs.keys()))
    all_persons.update(list(person_procs.keys()))
    all_persons = all_persons.difference(excluded_pid)
    all_persons = list(all_persons)

    # unique covariates
    all_covars = {v for inner_dict in person_conditions.values() for v in inner_dict}
    all_covars.update(v for inner_dict in person_drugs.values() for v in inner_dict)
    all_covars.update(v for inner_dict in person_procs.values() for v in inner_dict)
    all_covars = list(all_covars)
    print("total unique persons and unique covars: ", len(all_persons), len(all_covars))
    all_covars_idx = dict(zip(all_covars, range(len(all_covars))))
    print("number of diabetes cases without any conditions or drugs: ", len(set(list(p_diab_date_dict.keys())).difference(all_persons)))

    # generate CSR matrix and labels
    labels = []
    feature_matrix = lil_matrix((len(all_persons), len(all_covars)), dtype=np.uint16)
    for i, pid in enumerate(all_persons):
        labels.append(1 if pid in p_diab_date_dict else 0)  # update labels

        # condition covariates
        if pid in person_conditions:
            cond_covars = person_conditions[pid]
            for covar, val in cond_covars.items():
                j = all_covars_idx[covar]
                feature_matrix[i, j] = 65535 if val > 65535 else val

        # drug covariates
        if pid in person_drugs:
            drug_covars = person_drugs[pid]
            for covar, val in drug_covars.items():
                j = all_covars_idx[covar]
                feature_matrix[i, j] = 65535 if val > 65535 else val

        # procedure covariates
        if pid in person_procs:
            proc_covars = person_procs[pid]
            for covar, val in proc_covars.items():
                j = all_covars_idx[covar]
                feature_matrix[i, j] = 65535 if val > 65535 else val

        # keep track of progress
        if (i + 1) % 100000 == 0:
            print("CSR matrix generated for {0} persons".format(i + 1))
    print("CSR matrix generated for {0} persons".format(i + 1))

    return csr_matrix(feature_matrix), np.asarray(labels), np.asarray(all_persons), np.asarray(all_covars)


def regression_feature_and_label(person_conditions, person_drugs, person_obs, p_diab_date_dict, excluded_pid):
    """
    generate features in CSR format for ML
    """
    print("\ngenerate data for ML models")
    # unique persons
    all_persons = set(list(person_conditions.keys()))
    all_persons.update(list(person_drugs.keys()))
    all_persons = all_persons.difference(excluded_pid)
    all_persons = list(all_persons)

    # unique covariates
    all_covars = {v for inner_set in person_conditions.values() for v in inner_set}
    all_covars.update(v for inner_dict in person_drugs.values() for v in inner_dict)
    all_covars = list(all_covars)
    print("total unique persons and unique covars: ", len(all_persons), len(all_covars))
    all_covars_idx = dict(zip(all_covars, range(len(all_covars))))
    print("number of diabetes cases without any other condition and drugs: ", len(set(list(p_diab_date_dict.keys())).difference(all_persons)))

    # generate CSR matrix and labels
    labels = []
    feature_matrix = np.zeros((len(all_persons), len(all_covars)), dtype=np.uint8)
    for i, pid in enumerate(all_persons):
        obs_date = datetime.strptime(person_obs[pid], "%Y-%m-%d")
        diab_date = datetime.strptime(p_diab_date_dict[pid], "%Y-%m-%d")
        d = (diab_date - obs_date).days
        labels.append(d)  # update labels

        # condition covariates
        if pid in person_conditions:
            cond_covars = person_conditions[pid]
            for covar, val in cond_covars.items():
                j = all_covars_idx[covar]
                feature_matrix[i, j] = 255 if val > 255 else val

        # drug covariates
        if pid in person_drugs:
            drug_covars = person_drugs[pid]
            for covar, val in drug_covars.items():
                j = all_covars_idx[covar]
                feature_matrix[i, j] = 255 if val > 255 else val

        # keep track
        if (i + 1) % 100000 == 0:
            print("CSR matrix generated for {0} persons".format(i + 1))
    print("CSR matrix generated for {0} persons".format(i + 1))

    return csr_matrix(feature_matrix), np.asarray(labels), np.asarray(all_persons), np.asarray(all_covars)


def save_data_labels(X, y, persons, covariates, features_file=None, labels_file=None, persons_file=None, covars_file=None):
    """
    save data for future use as processing takes hours
    """
    # features
    with open(features_file, "wb") as fh:
        pickle.dump(X, fh, protocol=5)

    # labels
    with open(labels_file, "wb") as fh:
        pickle.dump(y, fh, protocol=5)

    # persons
    with open(persons_file, "wb") as fh:
        pickle.dump(persons, fh, protocol=5)

    # covariates
    with open(covars_file, "wb") as fh:
        pickle.dump(covariates, fh, protocol=5)


def str_to_bool(s):
    """
    convert command line True/False to boolean as argparse considers them string.
    """
    if s == 'True':
        return True
    elif s == 'False':
        return False
    else:
        print("invalid boolean value provided")
        return None


def main():
    """
    This program reads the SQL output files and generate feature matrix and labels for the ML models.
    The feature matrix and labels are saved as pickle files.
    """
    # load IO filenames from the YAML file
    parser = argparse.ArgumentParser()
    parser.add_argument("-iofiles", default="io_data.yaml", help="provide yaml file containing io files")
    p_args = parser.parse_args()
    with open(p_args.iofiles, 'r') as fi:
        iodata = yaml.safe_load(fi)

    # STEP 1: read concept file and create a dictionary with concept_id as key and vocab_id + concept_name as values
    concept_id_name_dict = fetch_concept_id_name(ifile=iodata['input_files']['concepts_file'])
    print("number of concepts: ", len(concept_id_name_dict))

    # STEP 2: create person and observation dates dictionary
    person_obs_dates_dict = None
    if not os.path.exists(iodata['output_files']['person_obs_date_file']):
        person_obs_dates_dict = fetch_observation_dates(ifile=iodata['input_files']['person_observation_file'],
                                                        ofile=iodata['output_files']['person_obs_date_file'])
    else:
        with open(iodata['output_files']['person_obs_date_file'], 'rb') as f:
            person_obs_dates_dict = pickle.load(f)
    print("unique patient observation count: ", len(person_obs_dates_dict))

    # STEP 3: fetch person details - age, sex, state
    person_data_dict = None
    if not os.path.exists(iodata['output_files']['person_data_file']):
        person_data_dict = get_person_details(person_obs_dates_dict, ifile=iodata['input_files']['person_data_file'],
                                              ofile=iodata['output_files']['person_data_file'])
    else:
        with open(iodata['output_files']['person_data_file'], 'rb') as f:
            person_data_dict = pickle.load(f)
    print("unique person count: ", len(person_data_dict.keys()))

    # STEP 4: select all type 1 and type 2 diabetes condition codes
    type1_diab_codes, type2_diab_codes = select_diabetes_codes(ifile1=iodata['input_files']['diab1_codes_file'],
                                                               ifile2=iodata['input_files']['diab2_codes_file'])
    print("number of diabetes condition codes (type1, type2): ", len(type1_diab_codes), len(type2_diab_codes))

    # STEP 5: create person and first diabetes code date dictionary
    person_diab_date_dict = None
    excluded_persons = None
    if not os.path.exists(iodata['output_files']['person_diab_date_file']):
        person_diab_date_dict, excluded_persons = fetch_first_diab_coding_date(type1_diab_codes, type2_diab_codes, person_obs_dates_dict,
                                                                               icd9_diab_codes=iodata['icd9_diab_codes'],
                                                                               ifile=iodata['input_files']['person_condition_file'],
                                                                               ofile1=iodata['output_files']['person_diab_date_file'],
                                                                               ofile2=iodata['output_files']['excluded_person_file'])
    else:
        with open(iodata['output_files']['person_diab_date_file'], 'rb') as f:
            person_diab_date_dict = pickle.load(f)
        with open(iodata['output_files']['excluded_person_file'], 'rb') as f:
            excluded_persons = pickle.load(f)
    print("unique type 1 diabetes patient count: ", len(excluded_persons))
    print("unique type 2 diabetes patient count: ", len(person_diab_date_dict))

    # STEP 6: fetch person conditions
    person_conditions_dict = None
    if not os.path.exists(iodata['output_files']['person_cond_file']):
        person_conditions_dict = get_person_conditions(type2_diab_codes, person_diab_date_dict, person_obs_dates_dict, concept_id_name_dict,
                                                       excluded_persons, ifile=iodata['input_files']['person_condition_file'],
                                                       ofile=iodata['output_files']['person_cond_file'],
                                                       covar_selection='before')
    else:
        with open(iodata['output_files']['person_cond_file'], 'rb') as f:
            person_conditions_dict = pickle.load(f)
    print("keys in person_conditions_dict: ", len(person_conditions_dict))

    # STEP 7: fetch person drugs
    person_drugs_dict = None
    if not os.path.exists(iodata['output_files']['person_drug_file']):
        person_drugs_dict = get_person_drugs(person_diab_date_dict, person_obs_dates_dict, excluded_persons,
                                             ifile=iodata['input_files']['person_drug_file'],
                                             ofile=iodata['output_files']['person_drug_file'],
                                             covar_selection='before')
    else:
        with open(iodata['output_files']['person_drug_file'], 'rb') as f:
            person_drugs_dict = pickle.load(f)
    print("keys in person_drugs_dict: ", len(person_drugs_dict))


    # STEP 8: fetch person procedures
    person_proc_dict = None
    '''
    if not os.path.exists(iodata['output_files']['person_proc_file']):
        person_proc_dict = get_person_procedures(person_diab_date_dict, person_obs_dates_dict, excluded_persons,
                                                 ifile=iodata['input_files']['person_procedure_file'],
                                                 ofile=iodata['output_files']['person_proc_file'])
    else:
        with open(iodata['output_files']['person_proc_file'], 'rb') as f:
            person_proc_dict = pickle.load(f)
    print("keys in person_proc_dict: ", len(person_proc_dict))
    '''

    # generate label and covariates for each person
    X_all, y_all, persons_all, covariates_all = generate_feature_and_label(person_diab_date_dict, excluded_persons,
                                                                           person_conditions=person_conditions_dict,
                                                                           person_drugs=person_drugs_dict,
                                                                           person_procs=person_proc_dict)

    print("len(persons_all), len(y_all), len(covariates_all): ", len(persons_all), len(y_all), len(covariates_all))
    print("persons with diabetes: ", np.sum(y_all), len(person_diab_date_dict))

    # save data, labels, covariates and person_ids
    save_data_labels(X_all, y_all, persons_all, covariates_all, features_file=iodata['output_files']['features_file'],
                     labels_file=iodata['output_files']['labels_file'], persons_file=iodata['output_files']['persons_file'],
                     covars_file=iodata['output_files']['covars_file'])


if __name__ == "__main__":
    main()
