#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 10:57:35 2023

@author: aubouinb
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from python_anesthesia_simulator import patient

# dataset carachteristic
sampling_time = 2
propo_concentration = 10  # mg/ml
remi_concentration = 50  # µg/ml

# import and process data
Patient_id = 2
Patient_1_df = pd.read_csv("./data/Patient_" + str(Patient_id) + "_data.csv")
Patient_1_df = Patient_1_df[['BIS1', 'InfRate_TT_prop', 'CeProp', 'CpProp', 'InfRate_TT_remi', 'CeRemi', 'CpRemi']]

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

istart = 0

for i in range(len(Patient_1_df)):
    if Patient_1_df.loc[i, 'InfRate_TT_prop'] > 0 or Patient_1_df.loc[i, 'InfRate_TT_remi'] > 0:
        istart = i
        break

# removed before strating of anesthesia and All the thing 10 minutes after
Patient_1_df = Patient_1_df[istart: int(istart + 30*60/sampling_time)]
Patient_1_df.reset_index(inplace=True)
Time = np.arange(0, len(Patient_1_df)*sampling_time, sampling_time)
# use mg/s for Propofol and µg/s for Remifentanil

Patient_1_df['InfRate_TT_prop'] = propo_concentration / 3600 * Patient_1_df['InfRate_TT_prop']
Patient_1_df['InfRate_TT_remi'] = remi_concentration / 3600 * Patient_1_df['InfRate_TT_remi']

Patient_simu = patient.Patient(age, height, weight, gender,
                               model_propo="Schnider", model_remi="Minto", Ts=sampling_time)


for i in range(len(Time)):
    Patient_simu.one_step(uP=Patient_1_df.loc[i, 'InfRate_TT_prop'], uR=Patient_1_df.loc[i, 'InfRate_TT_remi'])


fig, ax = plt.subplots(2)
ax[0].plot(Time, Patient_1_df['InfRate_TT_prop'])
ax[0].set_ylabel('Propo (mg/s)')
ax[0].grid()
ax[0].set_title('Input rate')
ax[1].plot(Time, Patient_1_df['InfRate_TT_remi'])
ax[1].set_ylabel('Remi (µg/s)')
plt.xlabel('time (s)')
plt.grid()
plt.show()


plt.plot(Time, Patient_simu.dataframe['x_propo_1'], label="simulation")
plt.plot(Time, Patient_1_df['CpProp'], label="data")
plt.title('Propofol plasma concentration')
plt.xlabel('time (s)')
plt.ylabel('Ce (µg/ml)')
plt.grid()
plt.legend()
plt.show()

plt.plot(Time, Patient_simu.dataframe['x_remi_1'], label="simulation")
plt.plot(Time, Patient_1_df['CpRemi'], label="data")
plt.title('Remifentanil plasma concentration')
plt.xlabel('time (s)')
plt.ylabel('Ce (ng/ml)')
plt.grid()
plt.legend()
plt.show()

plt.plot(Time, Patient_simu.dataframe['x_propo_4'], label="simulation")
plt.plot(Time, Patient_1_df['CeProp'], label="data")
plt.title('Propofol effect site concentration')
plt.xlabel('time (s)')
plt.ylabel('Ce (µg/ml)')
plt.grid()
plt.legend()
plt.show()

plt.plot(Time, Patient_simu.dataframe['x_remi_4'], label="simulation")
plt.plot(Time, Patient_1_df['CeRemi'], label="data")
plt.title('Remifentanil effect site concentration')
plt.xlabel('time (s)')
plt.ylabel('Ce (ng/ml)')
plt.grid()
plt.legend()
plt.show()
