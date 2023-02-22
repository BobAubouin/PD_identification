#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 18:41:43 2023

@author: aubouinb
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from python_anesthesia_simulator import patient
import casadi as cas

# dataset carachteristic
sampling_time = 2
propo_concentration = 10  # mg/ml
remi_concentration = 50  # µg/ml


# import and process data
Patient_id = 1
# Patient_1_df = pd.read_csv("./data/Patient_" + str(Patient_id) + "_data.csv")
# Patient_1_df = Patient_1_df[['BIS1', 'InfRate_TT_prop', 'CeProp', 'CpProp', 'InfRate_TT_remi', 'CeRemi', 'CpRemi']]
Patient_1_df = pd.read_csv("./data/raw/Patient_" + str(Patient_id) + "_data.csv")

# Patient 1
if Patient_id == 1:
    age = 31
    weight = 49
    height = 163
    gender = 0

# Patient 2
elif Patient_id == 2:
    age = 44
    weight = 57
    height = 168
    gender = 0

istart_propo = 0
istart_remi = 0
for i in range(len(Patient_1_df)):
    if Patient_1_df.loc[i, 'InfRate_TT_prop'] > 0 and istart_propo == 0:
        istart_propo = i
    if Patient_1_df.loc[i, 'InfRate_TT_remi'] > 0 and istart_remi == 0:
        istart_remi = i
    if istart_remi * istart_propo > 0:
        break
istart = min(istart_remi, istart_propo)

# removed before strating of anesthesia and All the thing 10 minutes after
Patient_1_df = Patient_1_df[istart:istart_remi]
Patient_1_df.reset_index(inplace=True)
Time = np.arange(0, len(Patient_1_df)*sampling_time, sampling_time)
# use mg/s for Propofol and µg/s for Remifentanil

Patient_1_df['InfRate_TT_prop'] = propo_concentration / 3600 * Patient_1_df['InfRate_TT_prop']
Patient_1_df['InfRate_TT_remi'] = remi_concentration / 3600 * Patient_1_df['InfRate_TT_remi']

Patient_1_df['InfRate_TT_prop_simu'] = propo_concentration / 3600 * Patient_1_df['InfRate_TT_prop_simu']
Patient_1_df['InfRate_TT_remi_simu'] = remi_concentration / 3600 * Patient_1_df['InfRate_TT_remi_simu']
Patient_1_df['CeProp'] = Patient_1_df['CeProp'] + 1e-3

Patient_simu = patient.Patient(age, height, weight, gender,
                               model_propo="Eleveld", model_remi="Eleveld", Ts=sampling_time)
Patient_simu_2 = patient.Patient(age, height, weight, gender,
                                 model_propo="Schnider", model_remi="Minto", Ts=sampling_time)

for i in range(len(Time)):
    Patient_simu.one_step(uP=Patient_1_df.loc[i, 'InfRate_TT_prop_simu'],
                          uR=Patient_1_df.loc[i, 'InfRate_TT_remi_simu'])

    Patient_simu_2.one_step(uP=Patient_1_df.loc[i, 'InfRate_TT_prop_simu'],
                            uR=Patient_1_df.loc[i, 'InfRate_TT_remi_simu'])


plt.plot(Time, Patient_simu.dataframe['x_propo_4'], label="Eleveld")
plt.plot(Time, Patient_simu_2.dataframe['x_propo_4'], label="Schnider")
plt.plot(Time, Patient_1_df['CeProp'], label="data")
plt.xlabel('time (s)')
plt.ylabel('Ce_propo ( µg/ml)')
plt.grid()
plt.legend()
plt.show()

# %% optimization problem
E0 = np.mean(Patient_1_df.loc[:5, 'BIS1'])

# SISO Problem
Emax = cas.MX.sym('Emax')
C50p = cas.MX.sym('C50p')
gamma = cas.MX.sym('gamma')
p = cas.MX.sym('ce_data', len(Time))
w = [Emax, C50p, gamma]
w0 = [Patient_simu.BisPD.Emax, Patient_simu.BisPD.c50p, Patient_simu.BisPD.gamma]
lbw = [70, 1, 1.01]
ubw = [100, 10, 10]
J = 0

for i in range(len(Time)):
    Ce = p[i]
    J += (cas.MX(Patient_1_df.loc[i, 'BIS1']) - (E0 - Emax * (Ce**gamma)/(C50p**gamma + Ce**gamma)))**2


opts = {'ipopt.print_level': 1, 'print_time': 0, 'ipopt.max_iter': 300}
prob = {'f': J, 'x': cas.vertcat(*w), 'p': p}
solver_siso = cas.nlpsol('solver', 'ipopt', prob)  # opts
sol = solver_siso(x0=w0, lbx=lbw, ubx=ubw, p=Patient_simu.dataframe['x_propo_4'].tolist())
sol_2 = solver_siso(x0=w0, lbx=lbw, ubx=ubw, p=Patient_simu_2.dataframe['x_propo_4'].tolist())
sol_3 = solver_siso(x0=w0, lbx=lbw, ubx=ubw, p=Patient_1_df['CeProp'].tolist())

w_opt = sol['x'].full().flatten()
w_opt_2 = sol_2['x'].full().flatten()
w_opt_3 = sol_3['x'].full().flatten()

print("Eleveld : ")
print(w_opt)
print("Schnider : ")
print(w_opt_2)
print("Schnider from data : ")
print(w_opt_3)

Emax = w_opt[0]
C50p = w_opt[1]
gamma = w_opt[2]

Emax_2 = w_opt_2[0]
C50p_2 = w_opt_2[1]
gamma_2 = w_opt_2[2]

Emax_3 = w_opt_3[0]
C50p_3 = w_opt_3[1]
gamma_3 = w_opt_3[2]


BIS = np.zeros(len(Time))
Ce = Patient_simu.dataframe['x_propo_4'].to_numpy()

BIS_2 = np.zeros(len(Time))
Ce_2 = Patient_simu_2.dataframe['x_propo_4'].to_numpy()

BIS_3 = np.zeros(len(Time))
Ce_3 = Patient_1_df['CeProp'].to_numpy()
for i in range(len(Time)):
    BIS[i] = (E0 - Emax * (Ce[i]**gamma)/(C50p**gamma + Ce[i]**gamma))

    BIS_2[i] = (E0 - Emax_2 * (Ce_2[i]**gamma_2)/(C50p_2**gamma_2 + Ce_2[i]**gamma_2))

    BIS_3[i] = (E0 - Emax_3 * (Ce_3[i]**gamma_3)/(C50p_3**gamma_3 + Ce_3[i]**gamma_3))

cost_E = np.sum(np.square(Patient_1_df['BIS1'].to_numpy() - BIS))
cost_S = np.sum(np.square(Patient_1_df['BIS1'].to_numpy() - BIS_2))

print('cost Eleveld :' + str(cost_E))
print('cost Schnider :' + str(cost_S))

plt.plot(Time, Patient_1_df['BIS1'], label='data')
plt.plot(Time, BIS, label="identified Eleveld")
plt.plot(Time, BIS_2, label="identified Schnider")
plt.plot(Time, BIS_3, label="identified data")
plt.xlabel('time (s)')
plt.ylabel('BIS')
plt.grid()
plt.legend()
plt.show()

# plt.plot(Patient_simu.dataframe['x_propo_4'], Patient_1_df['BIS1'], '*', label='data')
plt.plot(Ce, BIS, label="identified Eleveld")
plt.plot(Ce_2, BIS_2, label="identified Schnider")
plt.plot(Ce_3, BIS_3, label="identified data")
plt.xlabel('Effect site concentration (µg/ml)')
plt.ylabel('BIS')
plt.grid()
plt.legend()
plt.show()

plt.plot(Patient_simu_2.dataframe['x_propo_4'], Patient_1_df['BIS1'], '*', label='data')
plt.plot(Ce_2, BIS_2, label="identified")
plt.xlabel('Effect site concentration Schnider(µg/ml)')
plt.ylabel('BIS')
plt.grid()
plt.legend()
plt.show()

plt.plot(Patient_simu.dataframe['x_propo_4'], Patient_1_df['BIS1'], '*', label='data')
plt.plot(Ce, BIS, label="identified")
plt.xlabel('Effect site concentration Eleveld(µg/ml)')
plt.ylabel('BIS')
plt.grid()
plt.legend()
plt.show()
