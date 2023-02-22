"""
Created on Mon Feb 13 09:57:30 2023

@author: aubouinb
"""
# standard import
import os

# Third party imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import casadi as cas
from scipy.stats import pearsonr

# local import
from python_anesthesia_simulator import patient
from TCI_control import TCI


# %% define the outpur function of the Propo-Remi to BIS interaction
E0 = cas.MX.sym('E0')
Emax = cas.MX.sym('Emax')
C50p = cas.MX.sym('C50p')
C50r = cas.MX.sym('C50r')
gamma = cas.MX.sym('gamma')
beta = cas.MX.sym('beta')

xp = cas.MX.sym('cep')
xr = cas.MX.sym('cer')
Up = xp/C50p
Ur = xr/C50r
Phi = Up/(Up + Ur + 1e-6)
U_50 = 1 - beta * (Phi - Phi**2)
interaction = (Up + Ur)/U_50
BIS = E0 - Emax * interaction ** gamma / (1 + interaction ** gamma)

H = cas.Function('Output', [xp, xr, C50p, C50r, beta, gamma, Emax, E0], [BIS],
                 ['xp', 'xr', 'C50p', 'C50r', 'beta', 'gamma', 'Emax', 'E0'], ['bis'])

# %% start the case

plot = False  # use plot to display all the figure for each patient
p_value = 0.01
# dataset carachteristic
sampling_time = 5
propo_concentration = 10  # mg/ml
remi_concentration = 50  # µg/ml
Bad_data_id = [8, 26, 36, 37, 39, 57]  # Patient with missing data

data_folder_path = './data/raw/'
dir_list = os.listdir(data_folder_path)
patient_fiche = pd.read_csv('./data/patients_fiche.csv')

output_dataframe = pd.DataFrame()

count = 0
# Process each file i the folder "data"
for file_name in dir_list:
    if 'data' not in file_name and 'Patient' not in file_name:
        continue  # pass if the file name doesn't have the good format
    # get patient id from fil name
    _, Patient_id, _ = file_name.split("_", 2)
    Patient_id = int(Patient_id)
    if Patient_id in Bad_data_id:
        continue
    count += 1
    print('Patient ' + str(Patient_id) + ', ' + str(count) + '/' + str(70 - len(Bad_data_id)))
    Patient_df = pd.read_csv(data_folder_path + file_name)

    # patient information from the fiche
    age = int(patient_fiche.loc[patient_fiche['No.'] == Patient_id, 'Age'])
    height = int(patient_fiche.loc[patient_fiche['No.'] == Patient_id, 'Height'])
    weight = int(patient_fiche.loc[patient_fiche['No.'] == Patient_id, 'Weight'])
    gender = int(patient_fiche.loc[patient_fiche['No.'] == Patient_id, 'Sex'] == 'M')

    istart_propo = 0
    istart_remi = 0
    for i in range(len(Patient_df)):
        if Patient_df.loc[i, 'InfRate_TT_prop'] > 0 and istart_propo == 0:
            istart_propo = i
        if Patient_df.loc[i, 'InfRate_TT_remi'] > 0 and istart_remi == 0:
            istart_remi = i
        if istart_remi * istart_propo > 0:
            break
    istart = min(istart_remi, istart_propo)
    iend = max(istart_remi, istart_propo) + int(5*60/sampling_time)

    # removed before strating of anesthesia and All the thing 5 minutes after Remifentanil start
    Patient_df = Patient_df[istart:iend]
    Patient_df.reset_index(inplace=True)
    Time = np.arange(0, len(Patient_df)*sampling_time, sampling_time)

    # Fill missing_data
    Patient_df['CtProp'].fillna(method='bfill', inplace=True)
    Patient_df['CtRemi'].fillna(method='bfill', inplace=True)

    Patient_simu = patient.Patient(age, height, weight, gender,
                                   model_propo="Eleveld", model_remi="Eleveld", Ts=sampling_time)

    Propo_control = TCI([age, height, weight, gender], drug_name="Propofol",
                        model_used="Schnider", drug_concentration=propo_concentration, maximum_rate=1200)

    Remi_control = TCI([age, height, weight, gender], drug_name="Remifentanil",
                       model_used="Minto", drug_concentration=remi_concentration, maximum_rate=400)

    U_propo = np.zeros(len(Time))
    U_remi = np.zeros(len(Time))
    sampling_control = int(10/sampling_time)
    for i in range(len(Time)):
        if i % sampling_control == 0:
            if i >= (istart_propo - istart):
                U_propo[i:i+sampling_control] = Propo_control.one_step(target=Patient_df.loc[i, "CtProp"])
            if i >= (istart_remi - istart):
                U_remi[i:i+sampling_control] = Remi_control.one_step(target=Patient_df.loc[i, "CtRemi"])

        Patient_simu.one_step(uP=U_propo[i] * propo_concentration / 3600, uR=U_remi[i] * remi_concentration/3600)

    if plot:
        plt.plot(Time, Patient_simu.dataframe['x_propo_4'], label="Propo ( µg/ml)")
        plt.plot(Time, Patient_simu.dataframe['x_remi_4'], label="Remi ( ng/ml)")
        plt.plot(Time, Patient_df['CeProp'], label="data propo")
        plt.plot(Time, Patient_df['CeRemi'], label="data remi")
        plt.xlabel('time (s)')
        plt.ylabel('Ce')
        plt.grid()
        plt.legend()
        plt.title("Patient " + str(Patient_id))
        plt.show()

    # %% defines the problem
    BIS = Patient_df['BIS1'].to_numpy()
    BIS = np.sort(BIS)
    E0 = np.mean(BIS[-10:])

    J = 0
    delay = 0
    for i in range(len(Time)-delay):
        Cep = Patient_simu.dataframe.loc[i+delay, 'x_propo_4']
        Cer = Patient_simu.dataframe.loc[i+delay, 'x_remi_4']
        temp = H(xp=Cep, xr=Cer, C50p=C50p, C50r=C50r, beta=beta, gamma=gamma, Emax=Emax, E0=E0)
        J += (cas.MX(Patient_df.loc[i, 'BIS1']) - temp['bis'])**2

    # MISO Propofol and remifentnail to BIS
    opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.max_iter': 300}
    prob = {'f': J, 'x': cas.vertcat(*[C50p, C50r, gamma, beta, Emax])}
    solver_miso = cas.nlpsol('solver', 'ipopt', prob, opts)

    # %% solves the problem
    sol = solver_miso(x0=[Patient_simu.BisPD.c50p, Patient_simu.BisPD.c50r, Patient_simu.BisPD.gamma,
                          Patient_simu.BisPD.beta, Patient_simu.BisPD.Emax],
                      lbx=[1, 1, 1 + 1e-33, 0, 70],
                      ubx=[10, 40, 10, 4-1e-3, 100])
    w_opt = sol['x'].full().flatten()

    BIS = np.zeros(len(Time)-delay)
    BIS_data = np.zeros(len(Time)-delay)
    Cep = Patient_df['CeProp'].to_numpy()
    Cep = Patient_simu.dataframe['x_propo_4'].to_numpy()
    Cer = Patient_simu.dataframe['x_remi_4'].to_numpy()
    for i in range(len(Time)-delay):
        xp = Cep[i+delay]
        xr = 0
        temp = H(xp=xp, xr=xr, C50p=w_opt[0], C50r=w_opt[1], beta=w_opt[3], gamma=w_opt[2], Emax=w_opt[4], E0=E0)
        BIS[i] = float(temp['bis'])
        BIS_data[i] = Patient_df.loc[i, 'BIS1']
    chi_value, P_val = pearsonr(BIS_data, BIS)

    # %% save the results
    param_table = pd.DataFrame({'Patient_id': Patient_id, 'C50p': w_opt[0], 'C50r': w_opt[1],
                                'gamma': w_opt[2], 'beta': w_opt[3], 'E0': E0, 'Emax': w_opt[4], 'J': float(sol['f']),
                                'Chi_test': bool(P_val < p_value)},
                               index=[0])

    output_dataframe = pd.concat([output_dataframe, param_table], ignore_index=True)
    if plot:

        plt.plot(Time, Patient_df['BIS1'], label='data')
        plt.plot(Time, Patient_df['BIS1'], label='data')
        if delay > 0:
            plt.plot(Time[:-delay], BIS, label="identified")
        else:
            plt.plot(Time, BIS, label="identified")
        plt.xlabel('time (s)')
        plt.ylabel('BIS')
        plt.grid()
        plt.legend()
        plt.title("Patient " + str(Patient_id))
        plt.show()

        plt.plot(Patient_simu.dataframe['x_propo_4'], Patient_df['BIS1'], '*', label='data')
        plt.plot(Cep, BIS, label="identified")
        plt.xlabel('Effect site Propofol (µg/ml)')
        plt.ylabel('BIS')
        plt.grid()
        plt.legend()
        plt.title("Patient " + str(Patient_id))
        plt.show()

        plt.plot(Patient_simu.dataframe['x_remi_4'], Patient_df['BIS1'], '*', label='data')
        plt.plot(Cer, BIS, label="identified")
        plt.xlabel('Effect site Remifentanil (ng/ml)')
        plt.ylabel('BIS')
        plt.grid()
        plt.legend()
        plt.title("Patient " + str(Patient_id))
        plt.show()

# %% export results as csv
patient_fiche.rename(columns={'No.': 'Patient_id'}, inplace=True)
patient_fiche = patient_fiche[['Patient_id', 'Sex', 'Age', 'Height', 'Weight']]
patient_fiche.loc[:, 'Sex'] = [int(patient_fiche.loc[i, 'Sex'] == 'M') for i in range(len(patient_fiche['Sex']))]
output_dataframe = pd.merge(patient_fiche, output_dataframe, on='Patient_id')
output_dataframe = output_dataframe.round(2)
output_dataframe.to_csv("./outputs/datatable_propo_remi.csv")
