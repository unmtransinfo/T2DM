import argparse
import os
import yaml
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
from matplotlib.ticker import AutoMinorLocator
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def get_prediction_data(p_data_file=None, undiag_pred_file=None):
    """
    load saved data from pickle files
    """
    print("\nfetch prediction and demographics data from the file")
    if p_data_file is None or undiag_pred_file is None:
        print("please provide file name for prediction data")
        exit(0)
    else:
        # person demographics data
        with open(p_data_file, 'rb') as f:
            p_data_dict = pickle.load(f)

        # undiagnosed predictions
        with open(undiag_pred_file, 'rb') as f:
            undiag_pred_dict = pickle.load(f)

    return p_data_dict, undiag_pred_dict

def determine_undiagnosed_risk_categories(person_dem_info, undiagnosed_preds):
    """
    this function finds number of persons in each age group for low, moderate, and high risk categories
    """
    age_risk_dict = {}
    num, den = 0, 0
    print("populate the dictionary")
    for p_id, p_risk in undiagnosed_preds.items():
        p_age_group = person_dem_info[p_id][1]//10
        risk_percentile = p_risk[1]

        # data for >=18 age
        if person_dem_info[p_id][1] >= 18:
            den+=1
            if risk_percentile > 90:
                num+=1

        # initialize and update
        age_risk_dict.setdefault(p_age_group, [0, 0, 0])
        if risk_percentile < 50:
            age_risk_dict[p_age_group][0]+=1
        elif 50 <= risk_percentile <= 90:
            age_risk_dict[p_age_group][1]+=1
        elif risk_percentile > 90:
            age_risk_dict[p_age_group][2]+=1
    
    # print data for each age group
    for p_age, p_vals in age_risk_dict.items():
        total_count = sum(p_vals)
        print(f"{p_age*10}-{(p_age+1)*10-1}: {round(p_vals[0]*100/total_count,2)}, {round(p_vals[1]*100/total_count,2)}, {round(p_vals[2]*100/total_count,2)}, {total_count}")

    print(f"percentage of undiagnosed patients (>=18) with high risk: {round(num*100/den, 2)}")


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


def calculate_feature_counts(X, Y, covars, comorbidity_codes):
    """
    this function calculates the count and percentage for each important feature in diagnosed and undiagnosed patients
    """

    # positive and unlabeled counts
    X_pos = X[Y == 1]
    X_unlab = X[Y == 0]
    print(f"size of X_pos: {X_pos.shape}, size of X_unlab: {X_unlab.shape}")

    # count number of rows with non-zero value for each feature
    non_zero_count_pos = np.array((X_pos != 0).sum(axis=0)).flatten()  # Count of nonzero rows per feature
    non_zero_count_unlab = np.array((X_unlab != 0).sum(axis=0)).flatten()  # Count of nonzero rows per feature
    # print(f"non_zero_count_pos: {non_zero_count_pos}")
    # print(f"non_zero_count_unlab: {non_zero_count_unlab}")

    # compute percentage
    non_zero_frac_pos = non_zero_count_pos / X_pos.shape[0]
    non_zero_frac_unlab = non_zero_count_unlab / X_unlab.shape[0]

    for j, v in enumerate(covars):
        if v in comorbidity_codes:
            print(f"\ncoded: {comorbidity_codes[v]}: {non_zero_count_pos[j]}, {round(non_zero_frac_pos[j]*100, 2)}")
            print(f"unlabeled: {comorbidity_codes[v]}: {non_zero_count_unlab[j]}, {round(non_zero_frac_unlab[j]*100, 2)}")


def get_undiagnosed_count_percentile(undiag_pred_file=None):
    """
    load saved data from pickle files
    """
    percentile_count = {}
    print("\nfetch prediction data from the file")
    if undiag_pred_file is None:
        print("please provide file name for prediction data")
        exit(0)
    else:
        # undiagnosed predictions
        with open(undiag_pred_file, 'rb') as f:
            undiag_pred_dict = pickle.load(f)

    # find count for each percentile
    print("determine number of patients in each percentile group")
    for _, v in undiag_pred_dict.items():
        percentile_count[round(v[1], 0)] = percentile_count.get(round(v[1], 0), 0) + 1

    return percentile_count


def calculate_coded_fraction(p_data_file=None, person_list=None, labels_list=None, sex_file=None, age_file=None, state_file=None):
    """
    calculate the fraction of coded patients per state and age range
    """
    # load person list
    with open(person_list, 'rb') as f:
        pids = pickle.load(f)

    # load label list
    with open(labels_list, 'rb') as f:
        labels = pickle.load(f)
    print(f"number of patients in the list: {len(pids)}, {len(labels)}")

    # load patient info file
    with open(p_data_file, 'rb') as f:
        person_info = pickle.load(f)
    print(f"number of patients with demographic info: {len(person_info)}")

    # create dictionary for state and age range
    coded_sex_data = {}
    coded_age_data = {}
    coded_state_data = {}
    person_no_info_count = 0

    for j, p_id in enumerate(pids):
        if p_id not in person_info:
            person_no_info_count += 1
            continue
        # initialize and increase total count
        # sex
        coded_sex_data.setdefault('M' if person_info[p_id][0] else 'F', [0, 0]) # coded count, total count
        coded_sex_data['M' if person_info[p_id][0] else 'F'][1] += 1    # increase total count
        # age
        coded_age_data.setdefault(person_info[p_id][1]//10, [0, 0]) # coded count, total count
        coded_age_data[person_info[p_id][1] // 10][1] += 1 # increase total count
        # state
        coded_state_data.setdefault(person_info[p_id][2], [0, 0])   # coded count, total count
        coded_state_data[person_info[p_id][2]][1] += 1  # increase total count

        # increase coded count
        if labels[j] == 1:
            coded_sex_data['M' if person_info[p_id][0] else 'F'][0] += 1  # increase coded count
            coded_age_data[person_info[p_id][1] // 10][0] += 1  # increase coded count
            coded_state_data[person_info[p_id][2]][0] += 1  # increase coded count

        # keep track of the progress
        if (j+1)%2500000 == 0:
            print(f"number of patients processed: {j+1}")
    print(f"number of patients processed: {j + 1}, number of patients without info: {person_no_info_count}")

    # save dictionaries as pickle files
    with open(sex_file, 'wb') as f:  # sex data
        pickle.dump(coded_sex_data, f, protocol=5)

    with open(age_file, 'wb') as f:  # age data
        pickle.dump(coded_age_data, f, protocol=5)

    with open(state_file, 'wb') as f:  # state data
        pickle.dump(coded_state_data, f, protocol=5)

    return coded_sex_data, coded_age_data, coded_state_data

def plot_us_state_data(state_diab_data, plotFile=None):
    """
    Plot coded fraction data by state
    """
    # remove some of states data
    state_diab_data.pop('SC')
    state_diab_data.pop('UN')
    state_diab_data.pop('PR')

    # compute coded fraction for each state
    states = []
    fval = []
    for s, v in state_diab_data.items():
        states.append(s)
        fval.append(v[0] / (v[1]))

    df = pd.DataFrame({'state': states, 'fracs': fval})

    # create a choropleth map using plotly express
    fig = px.choropleth(
        df,
        locations='state',  # column in dataframe containing state abbreviations
        locationmode='USA-states',  # specify that we are plotting US states
        color='fracs',  # column in dataframe containing the average values
        scope='usa',  # limit map to the USA
        color_continuous_scale='rainbow',  # set the color scale
        # title='Coded fraction of opioid use disorder(OUD) by state'  # set the title
    )

    # add a legend for the color scale
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_colorbar=dict(
            title=dict(text='Coded Fraction', font=dict(size=18)),
            xanchor='right',
            tickfont=dict(size=16),
            x=1.10
        )
    )

    # save the map to a .png file
    pio.write_image(fig, plotFile, width=800, height=600, scale=4)


def plot_age_group_data(age_diab_data, plotFile=None):
    """
    generate bar plot using coded fraction for different age groups
    """
    age_group = []
    coded_fracs = []

    # compute fracs for each age group
    for s, v in age_diab_data.items():
        age_group.append(s)
        coded_fracs.append(v[0]/v[1])

    # sort by age group
    idx = np.argsort(age_group)
    age_group = np.asarray(age_group)[idx]
    coded_fracs = np.asarray(coded_fracs)[idx]
    xvals = [str(v * 10) + '-' + str((v + 1) * 10) for v in age_group]

    # create a bar plot
    fig = plt.figure(figsize=(2.5, 2.5))
    ax = fig.add_subplot(111)
    ax.bar(age_group, coded_fracs, width=0.5, color='crimson', align='center')

    # Set ticks and labels explicitly
    ax.set_xticks(age_group)
    ax.set_xticklabels(xvals, rotation=90)

    # Set minor ticks on y-axis
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    # set the x and y labels
    plt.xlabel('Age', fontsize=11)
    plt.ylabel('Coded fraction', fontsize=14)

    # show the plot
    plt.tight_layout()
    plt.savefig(plotFile, dpi=300, bbox_inches='tight', pad_inches=0.1)


def main():
    """
    This code contains functions for generating plots
    """
    comorbidity_codes = {45548653: "End stage renal disease",
                         45577516: "Secondary hyperparathyroidism of renal origin",
                         35207668: "Essential (primary) hypertension",
                         45552539: "Obstructive sleep apnea (adult) (pediatric)",
                         45586193: "Opioid dependence, uncomplicated",
                         35206695: "Iron deficiency anemia, unspecified",
                         35208968: "Cervicalgia",
                         35207065: "Hyperlipidemia, unspecified",
                         35206859: "Hypothyroidism, unspecified",
                         35207024: "Obesity, unspecified",
                         974166: "Hydrochlorothiazide"
                         }

    # load IO filenames from the YAML file
    parser = argparse.ArgumentParser()
    parser.add_argument("-iofiles", default="io_data.yaml", help="provide yaml file containing io files")
    p_args = parser.parse_args()
    with open(p_args.iofiles, 'r') as fi:
        iodata = yaml.safe_load(fi)

    # *** DETAILS 1 - START *** #

    # load predictions data for undiagnosed patients and demographics data of all patients
    person_dem_info, undiagnosed_preds = get_prediction_data(p_data_file=iodata['output_files']['person_data_file'],
                                                             undiag_pred_file=iodata['output_files']['uncoded_patients_posterior_file'])
    print(f"person demographics info: {len(person_dem_info)}, undiagnosed count: {len(undiagnosed_preds)}")

    # find the count and percentage of persons in each age group with low, moderate, and high risk 
    determine_undiagnosed_risk_categories(person_dem_info, undiagnosed_preds)

    # *** DETAILS 1 - END *** #
    exit(0)


    # load features, labels, covariates and person ids
    X_all, Y_all, rec_ids_all, covars_all = get_data_for_ml(features_file=iodata['output_files']['features_file'],
                                                            labels_file=iodata['output_files']['labels_file'],
                                                            persons_file=iodata['output_files']['persons_file'],
                                                            covars_file=iodata['output_files']['covars_file'])
    print(f"number of total records: {X_all.shape}, {len(Y_all)}, {len(rec_ids_all)}")
    print(f"number of covariates: {len(covars_all)},  number of diagnosed: {np.sum(Y_all)}")

    # count undiagnosed patients in each percentile
    undiagnosed_preds = get_undiagnosed_count_percentile(undiag_pred_file=iodata['output_files']['uncoded_patients_posterior_file'])
    for p_percentile, p_count in undiagnosed_preds.items():
        print(f"{p_percentile}: {p_count}")

    # compute prevalence of important features in diagnosed and undiagnosed patients
    calculate_feature_counts(X_all, Y_all, covars_all, comorbidity_codes)

    # select state data for the coded patients
    print("determine coded fraction by state and age")
    sex_plot_data, age_plot_data, state_plot_data = {}, {}, {}
    if not os.path.exists(iodata['plot_files']['state_plot_dict']):
        sex_plot_data, age_plot_data, state_plot_data = calculate_coded_fraction(p_data_file=iodata['output_files']['person_data_file'],
                                                                                 person_list=iodata['output_files']['persons_file'],
                                                                                 labels_list=iodata['output_files']['labels_file'],
                                                                                 sex_file=iodata['plot_files']['sex_plot_dict'],
                                                                                 age_file=iodata['plot_files']['age_plot_dict'],
                                                                                 state_file=iodata['plot_files']['state_plot_dict'])
    else:
        with open(iodata['plot_files']['sex_plot_dict'], 'rb') as f:    # sex data
            sex_plot_data = pickle.load(f)

        with open(iodata['plot_files']['age_plot_dict'], 'rb') as f:    # age data
            age_plot_data = pickle.load(f)

        with open(iodata['plot_files']['state_plot_dict'], 'rb') as f:  # state data
            state_plot_data = pickle.load(f)

    # print(sex_plot_data)
    # print(age_plot_data)
    # print(state_plot_data)

    # print sex and age data for Table 1
    pos_count = np.sum(Y_all)
    unlab_count = X_all.shape[0] - pos_count
    print(f"\ndiagnosed count: {pos_count}, undiagnosed count: {unlab_count}")
    
    # sex data
    for ss, vv in sex_plot_data.items():
        print(f"coded: {ss} -> {vv[0]}, {vv[0]*100/pos_count}")
        print(f"unlabeled: {ss} -> {vv[1]-vv[0]}, {(vv[1]-vv[0]) * 100 / unlab_count}")

    # age data
    for age, vv in age_plot_data.items():
        print(f"\ncoded: {age*10}-{(age+1)*10} -> {vv[0]}, {vv[0]*100/pos_count}")
        print(f"unlabeled: {age*10}-{(age+1)*10} -> {vv[1] - vv[0]}, {(vv[1] - vv[0]) * 100 / unlab_count}")

    # generate US map using coded data
    plot_us_state_data(state_plot_data, plotFile=iodata['plot_files']['diab_us_map_file'])

    # generate age group bar plot
    plot_age_group_data(age_plot_data, plotFile=iodata['plot_files']['diab_age_file'])

if __name__ == "__main__":
    main()
