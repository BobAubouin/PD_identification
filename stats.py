#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 16:52:23 2023

@author: aubouinb
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

patient_fiche = pd.read_csv('./data/patients_fiche.csv')
patient_fiche.rename(columns={'No.': 'Patient_id'}, inplace=True)
patient_fiche = patient_fiche[['Patient_id', 'Sex', 'Age', 'Height', 'Weight']]
patient_fiche['Sex'] = [int(patient_fiche.loc[i, 'Sex'] == 'M') for i in range(len(patient_fiche['Sex']))]
patient_PD = pd.read_csv('./outputs/datatable.csv')

data = pd.merge(patient_fiche, patient_PD, on='Patient_id')

corr_matrix = data.corr().round(2)
sns.heatmap(corr_matrix[6:][['Sex', 'Age', 'Height', 'Weight']], annot=True)
sns.set(rc={"figure.figsize": (10, 10)})
plt.show()
