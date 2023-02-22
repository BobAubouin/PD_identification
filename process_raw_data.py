"""
Created on Tue Feb 21 11:03:19 2023

@author: aubouinb
"""

# standard import
import os
# Third party imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# local import
from python_anesthesia_simulator import patient
from TCI_control import TCI


def read_time(string: str) -> int:
    """Convert string time to int.

    Parameters
    ----------
    string : str
        format = "HH:MM:SS".

    Returns
    -------
    int
        time in seconds.
    """
    hr, mi, se = string.split(":", 3)
    return int(hr) * 3600 + int(mi)*60 + int(se)


# dataset carachteristic
sampling_time = 5
propo_concentration = 10  # mg/ml
remi_concentration = 50  # µg/ml

data_folder_path = './data/raw/'
logfile_folder_path = './data/LogFiles/'
data_folder_output = './data/propo_induction/'
dir_list = os.listdir(data_folder_path)
patient_fiche = pd.read_csv('./data/patients_fiche.csv')

plot = True  # bool to actvate plot in the script
count = 0
count_bug = 0
# Process each file i the folder "data"
for file_name in dir_list:
    if 'data' not in file_name and 'Patient' not in file_name:
        continue  # pass if the file name doesn't have the good format
    # get patient id from fil name
    _, Patient_id, _ = file_name.split("_", 2)
    Patient_id = int(Patient_id)
    # if Patient_id not in [5, 6, 9, 21]:
    #     continue
    count += 1
    # if Patient_id != 30:
    #     continue
    # if count == 2:
    #     break
    print('Patient ' + str(Patient_id) + ', ' + str(count) + '/70')
    Patient_df = pd.read_csv(data_folder_path + file_name)
    Patient_df['CaseTime_TT_prop'] = Patient_df['CaseTime_TT_prop'].map(read_time)
    Patient_df['CaseTime_TT_bis'] = Patient_df['CaseTime_TT_bis'].map(read_time)

    logfile_df = pd.read_csv(logfile_folder_path + 'patient_' + str(Patient_id) + '.txt', sep=':', header=None,
                             names=['Name', 'hours', 'minutes', 'seconds'])

    # patient information from the fiche
    age = int(patient_fiche.loc[patient_fiche['No.'] == Patient_id, 'Age'])
    height = int(patient_fiche.loc[patient_fiche['No.'] == Patient_id, 'Height'])
    weight = int(patient_fiche.loc[patient_fiche['No.'] == Patient_id, 'Weight'])
    gender = int(patient_fiche.loc[patient_fiche['No.'] == Patient_id, 'Sex'] == 'M')

    istart_propo = 0
    istart_remi = 0
    for i in range(len(Patient_df)):
        if Patient_id == 30:
            if Patient_df.loc[i, 'InfRate_TT_prop'] > 0 and istart_propo == 0:
                istart_propo = i
        else:
            if Patient_df.loc[i, 'CeProp'] > 0 and istart_propo == 0:
                istart_propo = i
        if Patient_df.loc[i, 'InfRate_TT_remi'] > 0 and istart_remi == 0:
            istart_remi = i
        if istart_remi * istart_propo > 0:
            break

    istart = min(istart_remi, istart_propo)
    tof_start_time = int((int(logfile_df[logfile_df['Name'] == 'TOF_MON1']['hours']) * 3600 +
                          int(logfile_df[logfile_df['Name'] == 'TOF_MON1']['minutes']) * 60 +
                          int(logfile_df[logfile_df['Name'] == 'TOF_MON1']['seconds'])))

    for i in range(len(Patient_df)):
        if Patient_df.loc[i, 'CaseTime_TT_bis'] >= tof_start_time:
            i_tof = i-1
            break
    iend = min(istart_remi, i_tof)

    # removed before strating of anesthesia and All the thing 5 minutes after Remifentanil start
    Patient_df = Patient_df[istart:iend]
    Patient_df.reset_index(inplace=True)
    Time = np.arange(0, len(Patient_df)*sampling_time, sampling_time)

    # Fill missing_data
    Patient_df['CtProp'].fillna(method='bfill', inplace=True)
    Patient_df['CtRemi'].fillna(method='bfill', inplace=True)

    Patient_simu = patient.Patient(age, height, weight, gender,
                                   model_propo='Eleveld', Ts=sampling_time)

    Patient_simu_2 = patient.Patient(age, height, weight, gender,
                                     model_propo='Eleveld', Ts=sampling_time)

    Patient_simu_3 = patient.Patient(age, height, weight, gender,
                                     model_propo='Schnider', Ts=sampling_time)

    Propo_control = TCI([age, height, weight, gender], drug_name="Propofol",
                        model_used='Schnider', drug_concentration=propo_concentration, maximum_rate=1200)

    U_propo = np.zeros(len(Time))
    sampling_control = int(10/sampling_time)
    for i in range(len(Time)):
        if i % sampling_control == 0:
            if i >= (istart_propo - istart):
                U_propo[i:i+sampling_control] = Propo_control.one_step(target=Patient_df.loc[i, "CtProp"])

        Patient_simu.one_step(uP=U_propo[i] * propo_concentration / 3600)
        Patient_simu_3.one_step(uP=U_propo[i] * propo_concentration / 3600)

        up_raw = Patient_df.loc[i, 'InfRate_TT_prop']
        Patient_simu_2.one_step(uP=up_raw * propo_concentration / 3600)

    Patient_df.insert(len(Patient_df.columns), 'Upropo_simu', U_propo)
    Patient_df.insert(len(Patient_df.columns), 'Ce_Eleveld_raw', Patient_simu_2.dataframe['x_propo_4'].to_numpy())
    Patient_df.insert(len(Patient_df.columns), 'Ce_Eleveld_simu', Patient_simu.dataframe['x_propo_4'].to_numpy())
    Patient_df.insert(len(Patient_df.columns), 'Ce_Schnider_simu', Patient_simu_3.dataframe['x_propo_4'].to_numpy())
    Patient_df['Time'] = Time
    Patient_df = Patient_df[['Time', 'InfRate_TT_prop', 'Upropo_simu', 'BIS1', 'CeProp', 'Ce_Eleveld_raw',
                             'Ce_Eleveld_simu', 'Ce_Schnider_simu']]
    Patient_df.to_csv(data_folder_output + file_name)

    if plot:
        plt.plot(Time, Patient_df['CeProp'], label='raw')
        plt.plot(Time, Patient_df['Ce_Eleveld_raw'], label='Eleveld from raw inputs')
        plt.plot(Time, Patient_df['Ce_Eleveld_simu'], label='Eleveld from simulated inputs')
        plt.plot(Time, Patient_df['Ce_Schnider_simu'], label='Schnider from simulated inputs')
        plt.ylabel('Effect site Propofol (µg/ml)')
        plt.ylabel('Time (s)')
        plt.grid()
        plt.legend()
        plt.title("Patient " + str(Patient_id))
        plt.show()
