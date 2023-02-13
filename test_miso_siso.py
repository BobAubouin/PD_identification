#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 09:57:30 2023

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
Patient_id = 2
# Patient_1_df = pd.read_csv("./data/Patient_" + str(Patient_id) + "_data.csv")
# Patient_1_df = Patient_1_df[['BIS1', 'InfRate_TT_prop', 'CeProp', 'CpProp', 'InfRate_TT_remi', 'CeRemi', 'CpRemi']]
Patient_1_df = pd.read_csv("./data/Patien_" + str(Patient_id) + "_induction.csv")

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
Patient_1_df = Patient_1_df[istart:istart_remi + int(5*60/sampling_time)]
Patient_1_df.reset_index(inplace=True)
Time = np.arange(0, len(Patient_1_df)*sampling_time, sampling_time)
# use mg/s for Propofol and µg/s for Remifentanil

Patient_1_df['InfRate_TT_prop'] = propo_concentration / 3600 * Patient_1_df['InfRate_TT_prop']
Patient_1_df['InfRate_TT_remi'] = remi_concentration / 3600 * Patient_1_df['InfRate_TT_remi']

Patient_1_df['InfRate_TT_prop_simu'] = propo_concentration / 3600 * Patient_1_df['InfRate_TT_prop_simu']
Patient_1_df['InfRate_TT_remi_simu'] = remi_concentration / 3600 * Patient_1_df['InfRate_TT_remi_simu']

Patient_simu = patient.Patient(age, height, weight, gender,
                               model_propo="Schnider", model_remi="Minto", Ts=sampling_time)

for i in range(len(Time)):
    Patient_simu.one_step(uP=Patient_1_df.loc[i, 'InfRate_TT_prop_simu'],
                          uR=Patient_1_df.loc[i, 'InfRate_TT_remi_simu'])


plt.plot(Time, Patient_simu.dataframe['x_propo_4'], label="Propo ( µg/ml)")
plt.plot(Time, Patient_simu.dataframe['x_remi_4'], label="Remi ( ng/ml)")
plt.plot(Time, Patient_1_df['CeProp'], label="data")
plt.xlabel('time (s)')
plt.ylabel('Ce')
plt.grid()
plt.legend()
plt.show()

# %% defines the problems
E0 = np.mean(Patient_1_df.loc[:5, 'BIS1'])

Emax = cas.MX.sym('Emax')
C50p = cas.MX.sym('C50p')
C50r = cas.MX.sym('C50r')
gamma = cas.MX.sym('gamma')
beta = cas.MX.sym('beta')
Cep_siso = cas.MX.sym('cep_data', istart_remi - istart)
Cep_siso_2 = cas.MX.sym('cep_data', len(Time) - (istart_remi - istart))
Cer_siso_2 = cas.MX.sym('cer_data', len(Time) - (istart_remi - istart))
Cep = cas.MX.sym('cep_data', len(Time))
Cer = cas.MX.sym('cer_data', len(Time))

xp = cas.MX.sym('cep')
xr = cas.MX.sym('cer')
Up = xp/C50p
Ur = xr/C50r
Phi = Up/(Up + Ur + 1e-6)
U_50 = 1 - beta * (Phi - Phi**2)
interaction = (Up + Ur)/U_50
BIS = E0 - Emax * interaction ** gamma / (1 + interaction ** gamma)

H = cas.Function('Output', [xp, xr, C50p, C50r, beta, gamma, Emax], [BIS],
                 ['xp', 'xr', 'C50p', 'C50r', 'beta', 'gamma', 'Emax'], ['bis'])


J_siso = 0
J_siso_2 = 0
J_miso = 0
for i in range(len(Time)):
    if i < istart_remi - istart:
        Ce = Cep_siso[i]
        temp = H(xp=Ce, xr=0, C50p=C50p, C50r=1, beta=0, gamma=gamma, Emax=Emax)
        J_siso += (cas.MX(Patient_1_df.loc[i, 'BIS1']) - temp['bis'])**2
    else:
        xp = Cep_siso_2[i - (istart_remi - istart)]
        xr = Cer_siso_2[i - (istart_remi - istart)] + 1e-3
        temp = H(xp=xp, xr=xr, C50p=C50p, C50r=C50r, beta=beta, gamma=gamma, Emax=Emax)
        J_siso_2 += (cas.MX(Patient_1_df.loc[i, 'BIS1']) - temp['bis'])**2

    temp = H(xp=Cep[i], xr=Cer[i], C50p=C50p, C50r=C50r, beta=beta, gamma=gamma, Emax=Emax)
    J_miso += (cas.MX(Patient_1_df.loc[i, 'BIS1']) - temp['bis'])**2

# SISO part1 Propofol to BIS
opts = {'ipopt.print_level': 1, 'print_time': 0, 'ipopt.max_iter': 300}
prob = {'f': J_siso, 'x': cas.vertcat(*[Emax, C50p, gamma]), 'p': Cep_siso}
solver_siso = cas.nlpsol('solver', 'ipopt', prob)  # opts

# SISO part 2 Remifentanil to BIS
opts = {'ipopt.print_level': 2, 'print_time': 0, 'ipopt.max_iter': 300}
prob = {'f': J_siso_2, 'x': cas.vertcat(*[C50r, beta]),
        'p': cas.vertcat(*[Cep_siso_2, Cer_siso_2, Emax, C50p, gamma])}
solver_siso_2 = cas.nlpsol('solver', 'ipopt', prob)  # opts

# MISO Propofol and remifentnail to BIS
opts = {'ipopt.print_level': 1, 'print_time': 0, 'ipopt.max_iter': 300}
prob = {'f': J_miso, 'x': cas.vertcat(*[Emax, C50p, gamma, C50r, beta]), 'p': cas.vertcat(*[Cep, Cer])}
solver_miso = cas.nlpsol('solver', 'ipopt', prob)  # opts


# %% solves the problems

sol = solver_siso(x0=[Patient_simu.BisPD.Emax, Patient_simu.BisPD.c50p, Patient_simu.BisPD.gamma],
                  lbx=[70, 1, 1.01],
                  ubx=[100, 10, 10],
                  p=Patient_simu.dataframe.loc[:istart_remi - istart-1, 'x_propo_4'].tolist())

w_opt = sol['x'].full().flatten()


print("sol siso 1 : ")
print(w_opt)


Emax = w_opt[0]
C50p = w_opt[1]
gamma = w_opt[2]

sol = solver_siso_2(x0=[Patient_simu.BisPD.c50r, Patient_simu.BisPD.beta],
                    lbx=[1, 0],
                    ubx=[25, 4-1e-3],
                    p=[*Patient_simu.dataframe.loc[istart_remi - istart:, 'x_propo_4'].tolist(),
                       *Patient_simu.dataframe.loc[istart_remi - istart:, 'x_remi_4'].tolist(),
                       Emax, C50p, gamma])
w_opt = sol['x'].full().flatten()
print("sol siso 2 : ")
print(w_opt)


C50r = w_opt[0]
beta = w_opt[1]

sol = solver_miso(x0=[Patient_simu.BisPD.Emax, Patient_simu.BisPD.c50p, Patient_simu.BisPD.gamma,
                      Patient_simu.BisPD.c50r, Patient_simu.BisPD.beta],
                  lbx=[70, 1, 1.01, 2, 0],
                  ubx=[100, 10, 10, 25, 4-1e-3],
                  p=[*Patient_simu.dataframe.loc[:, 'x_propo_4'].tolist(),
                     *Patient_simu.dataframe.loc[:, 'x_remi_4'].tolist()])
w_opt = sol['x'].full().flatten()
Emax_miso = w_opt[0]
C50p_miso = w_opt[1]
gamma_miso = w_opt[2]
C50r_miso = w_opt[3]
beta_miso = w_opt[4]

param_table = pd.DataFrame({'Patient_id': Patient_id, 'C50p': C50p, 'C50r': C50r,
                            'gamma': gamma, 'beta': beta, 'E0': E0, 'Emax': Emax},
                           index=[0])
print("SISO param")
print(param_table)

param_table_miso = pd.DataFrame({'Patient_id': Patient_id, 'C50p': C50p_miso, 'C50r': C50r_miso,
                                 'gamma': gamma_miso, 'beta': beta_miso, 'E0': E0, 'Emax': Emax_miso},
                                index=[0])
print("MISO param")
print(param_table_miso)

BIS = np.zeros(len(Time))
Cep = Patient_simu.dataframe['x_propo_4'].to_numpy()
Cer = Patient_simu.dataframe['x_remi_4'].to_numpy()
BIS_2 = np.zeros(len(Time))
for i in range(len(Time)):
    xp = Cep[i]
    xr = Cer[i]

    temp = H(xp=xp, xr=xr, C50p=C50p, C50r=C50r, beta=beta, gamma=gamma, Emax=Emax)
    BIS[i] = float(temp['bis'])

    temp = H(xp=xp, xr=xr, C50p=C50p_miso, C50r=C50r_miso, beta=beta_miso, gamma=gamma_miso, Emax=Emax_miso)
    BIS_2[i] = float(temp['bis'])


cost_siso = np.sum(np.square(Patient_1_df['BIS1'].to_numpy() - BIS))
cost_miso = np.sum(np.square(Patient_1_df['BIS1'].to_numpy() - BIS_2))

print('cost SISO :' + str(cost_siso))
print('cost MISO :' + str(cost_miso))

plt.plot(Time, Patient_1_df['BIS1'], label='data')
plt.plot(Time, BIS, label="identified SISO")
plt.plot(Time, BIS_2, label="identified MISO")
plt.xlabel('time (s)')
plt.ylabel('BIS')
plt.grid()
plt.legend()
plt.show()


plt.plot(Patient_simu.dataframe['x_propo_4'], Patient_1_df['BIS1'], '*', label='data')
plt.plot(Cep, BIS, label="SISO")
plt.plot(Cep, BIS_2, label="MISO")
plt.xlabel('Effect site concentration Eleveld(µg/ml)')
plt.ylabel('BIS')
plt.grid()
plt.legend()
plt.show()
