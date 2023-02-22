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
from scipy.stats import truncnorm, lognorm
from os.path import exists

# %% theoretical value


def truncated_normal(mean=0, sd=1, low=0, upp=10):
    """Generate a random float number from a a truncate dnormal distribution."""
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


C50p_x = np.linspace(0, 20, 100)
C50p_y = truncated_normal(mean=4.47, sd=4.47**0.182, low=1, upp=10).pdf(C50p_x)
c50r_x = np.linspace(0, 40, 100)
c50r_y = truncated_normal(mean=19.3, sd=19.3*0.888, low=1, upp=40).pdf(c50r_x)
beta_x = np.linspace(-0.1, 3, 100)
beta_y = truncated_normal(mean=0, sd=0.5, low=0, upp=3).pdf(beta_x)
gamma_x = np.linspace(0, 10, 100)
gamma_y = truncated_normal(mean=1.43, sd=1.43*0.3, low=1, upp=5).pdf(gamma_x)
E0_x = np.linspace(80, 100, 100)
E0_y = truncated_normal(mean=97.4, sd=97.4*0.05, low=80, upp=100).pdf(E0_x)


# %% Propo - Remi Emax free
if exists('./outputs/datatable_propo_remi.csv'):
    data = pd.read_csv('./outputs/datatable_propo_remi.csv')

    corr_matrix = data.corr().round(2)
    sns.heatmap(corr_matrix[6:][['Sex', 'Age', 'Height', 'Weight']], annot=True)
    sns.set(rc={"figure.figsize": (10, 10)})
    plt.show()
    sns.set(style="ticks")

    ax = data[['C50p', 'C50r', 'gamma', 'beta', 'Emax', 'E0', 'J']].hist(bins=20)  #

    ax.flatten()[0].plot(C50p_x, C50p_y/max(C50p_y)*ax.flatten()[0].get_ylim()[1], 'r')
    ax.flatten()[1].plot(c50r_x, c50r_y/max(c50r_y)*ax.flatten()[1].get_ylim()[1], 'r')
    ax.flatten()[2].plot(gamma_x, gamma_y/max(gamma_y)*ax.flatten()[2].get_ylim()[1], 'r')
    ax.flatten()[3].plot(beta_x, beta_y/max(beta_y)*ax.flatten()[3].get_ylim()[1], 'r')
    ax.flatten()[4].plot(E0_x, E0_y/max(E0_y)*ax.flatten()[4].get_ylim()[1], 'r')
    ax.flatten()[5].plot(E0_x, E0_y/max(E0_y)*ax.flatten()[5].get_ylim()[1], 'r')

    fig = plt.gcf()
    fig.suptitle('All patient')
    plt.show()

    data = data[data['Chi_test']]
    data.reset_index(inplace=True)
    ax = data[['C50p', 'C50r', 'gamma', 'beta', 'Emax', 'E0', 'J']].hist(bins=20)  #

    ax.flatten()[0].plot(C50p_x, C50p_y/max(C50p_y)*ax.flatten()[0].get_ylim()[1], 'r')
    ax.flatten()[1].plot(c50r_x, c50r_y/max(c50r_y)*ax.flatten()[1].get_ylim()[1], 'r')
    ax.flatten()[2].plot(gamma_x, gamma_y/max(gamma_y)*ax.flatten()[2].get_ylim()[1], 'r')
    ax.flatten()[3].plot(beta_x, beta_y/max(beta_y)*ax.flatten()[3].get_ylim()[1], 'r')
    ax.flatten()[4].plot(E0_x, E0_y/max(E0_y)*ax.flatten()[4].get_ylim()[1], 'r')
    ax.flatten()[5].plot(E0_x, E0_y/max(E0_y)*ax.flatten()[5].get_ylim()[1], 'r')
    fig = plt.gcf()
    fig.suptitle('Signifiant patient')
    plt.show()

    plt.figure()
    Cep = np.linspace(0, 10)
    BIS = np.zeros((len(data), len(Cep)))
    for i in range(len(data)):
        for j in range(len(Cep)):
            Up = Cep[j]/data.loc[i, 'C50p']
            BIS[i, j] = data.loc[i, 'E0'] - data.loc[i, 'Emax'] * \
                Up ** data.loc[i, 'gamma'] / (1 + Up ** data.loc[i, 'gamma'])

        plt.plot(Cep, BIS[i, :], linewidth=0.5)

    plt.xlabel('Effect site Propofol (µg/ml)')
    plt.ylabel('BIS')
    plt.grid()
    plt.show()

# %% Propo - Remi Emax fixed
if exists('./outputs/datatable_propo_remi_Emax=E0.csv'):
    data = pd.read_csv('./outputs/datatable_propo_remi_Emax=E0.csv')

    corr_matrix = data.corr().round(2)
    sns.heatmap(corr_matrix[6:][['Sex', 'Age', 'Height', 'Weight']], annot=True)
    sns.set(rc={"figure.figsize": (10, 10)})
    plt.show()
    sns.set(style="ticks")

    ax = data[['C50p', 'C50r', 'gamma', 'beta', 'Emax', 'E0', 'J']].hist(bins=20)  #

    ax.flatten()[0].plot(C50p_x, C50p_y/max(C50p_y)*ax.flatten()[0].get_ylim()[1], 'r')
    ax.flatten()[1].plot(c50r_x, c50r_y/max(c50r_y)*ax.flatten()[1].get_ylim()[1], 'r')
    ax.flatten()[2].plot(gamma_x, gamma_y/max(gamma_y)*ax.flatten()[2].get_ylim()[1], 'r')
    ax.flatten()[3].plot(beta_x, beta_y/max(beta_y)*ax.flatten()[3].get_ylim()[1], 'r')
    ax.flatten()[4].plot(E0_x, E0_y/max(E0_y)*ax.flatten()[4].get_ylim()[1], 'r')
    ax.flatten()[5].plot(E0_x, E0_y/max(E0_y)*ax.flatten()[5].get_ylim()[1], 'r')

    fig = plt.gcf()
    fig.suptitle('All patient')
    plt.show()

    data = data[data['Chi_test']]
    data.reset_index(inplace=True)
    ax = data[['C50p', 'C50r', 'gamma', 'beta', 'Emax', 'E0', 'J']].hist(bins=20)  #

    ax.flatten()[0].plot(C50p_x, C50p_y/max(C50p_y)*ax.flatten()[0].get_ylim()[1], 'r')
    ax.flatten()[1].plot(c50r_x, c50r_y/max(c50r_y)*ax.flatten()[1].get_ylim()[1], 'r')
    ax.flatten()[2].plot(gamma_x, gamma_y/max(gamma_y)*ax.flatten()[2].get_ylim()[1], 'r')
    ax.flatten()[3].plot(beta_x, beta_y/max(beta_y)*ax.flatten()[3].get_ylim()[1], 'r')
    ax.flatten()[4].plot(E0_x, E0_y/max(E0_y)*ax.flatten()[4].get_ylim()[1], 'r')
    ax.flatten()[5].plot(E0_x, E0_y/max(E0_y)*ax.flatten()[5].get_ylim()[1], 'r')
    fig = plt.gcf()
    fig.suptitle('Signifiant patient')
    plt.show()

    plt.figure()
    Cep = np.linspace(0, 10)
    BIS = np.zeros((len(data), len(Cep)))
    for i in range(len(data)):
        for j in range(len(Cep)):
            Up = Cep[j]/data.loc[i, 'C50p']
            BIS[i, j] = data.loc[i, 'E0'] - data.loc[i, 'Emax'] * \
                Up ** data.loc[i, 'gamma'] / (1 + Up ** data.loc[i, 'gamma'])

        plt.plot(Cep, BIS[i, :], linewidth=0.5)

    plt.xlabel('Effect site Propofol (µg/ml)')
    plt.ylabel('BIS')
    plt.grid()
    plt.show()

# %% Propo
if exists('./outputs/datatable_propo.csv'):
    data = pd.read_csv('./outputs/datatable_propo.csv')

    corr_matrix = data.corr().round(2)
    sns.heatmap(corr_matrix[6:][['Sex', 'Age', 'Height', 'Weight']], annot=True)
    sns.set(rc={"figure.figsize": (10, 10)})
    plt.show()
    sns.set(style="ticks")

    ax = data[['C50p', 'gamma', 'Emax', 'E0', 'J']].hist(bins=20)
    ax.flatten()[0].plot(C50p_x, C50p_y/max(C50p_y)*ax.flatten()[0].get_ylim()[1], 'r')
    ax.flatten()[1].plot(gamma_x, gamma_y/max(gamma_y)*ax.flatten()[1].get_ylim()[1], 'r')
    ax.flatten()[2].plot(E0_x, E0_y/max(E0_y)*ax.flatten()[2].get_ylim()[1], 'r')
    ax.flatten()[3].plot(E0_x, E0_y/max(E0_y)*ax.flatten()[3].get_ylim()[1], 'r')
    fig = plt.gcf()
    fig.suptitle('All patient')
    plt.show()  # , 'J'

    data = data[data['Chi_test']]
    data.reset_index(inplace=True)

    ax = data[['C50p', 'gamma', 'Emax', 'E0']].hist(bins=20)
    ax.flatten()[0].plot(C50p_x, C50p_y/max(C50p_y)*ax.flatten()[0].get_ylim()[1], 'r')
    ax.flatten()[1].plot(gamma_x, gamma_y/max(gamma_y)*ax.flatten()[1].get_ylim()[1], 'r')
    ax.flatten()[2].plot(E0_x, E0_y/max(E0_y)*ax.flatten()[2].get_ylim()[1], 'r')
    ax.flatten()[3].plot(E0_x, E0_y/max(E0_y)*ax.flatten()[3].get_ylim()[1], 'r')
    fig = plt.gcf()
    fig.suptitle('Signifiant patient')
    plt.show()

    plt.figure()

    sns.set(style="ticks")
    Cep = np.linspace(0, 10)
    BIS = np.zeros((len(data), len(Cep)))
    for i in range(len(data)):
        for j in range(len(Cep)):
            Up = Cep[j]/data.loc[i, 'C50p']
            BIS[i, j] = data.loc[i, 'E0'] - data.loc[i, 'Emax'] * \
                Up ** data.loc[i, 'gamma'] / (1 + Up ** data.loc[i, 'gamma'])

        plt.plot(Cep, BIS[i, :], linewidth=1)

    plt.xlabel('Effect site Propofol (µg/ml)')
    plt.ylabel('BIS')
    plt.grid()
    plt.show()

# %% Propo Emax fixed

if exists('./outputs/datatable_propo_Ema=E0.csv'):
    data = pd.read_csv('./outputs/datatable_propo_Emax_trial.csv')

    corr_matrix = data.corr().round(2)
    sns.heatmap(corr_matrix[6:-3][['Sex', 'Age', 'Height', 'Weight']], annot=True)
    sns.set(rc={"figure.figsize": (10, 10)})
    plt.show()
    sns.set(style="ticks")

    ax = data[['C50p', 'gamma', 'Emax', 'E0', 'J']].hist(bins=20)
    ax.flatten()[0].plot(C50p_x, C50p_y/max(C50p_y)*ax.flatten()[0].get_ylim()[1], 'r')
    ax.flatten()[1].plot(gamma_x, gamma_y/max(gamma_y)*ax.flatten()[1].get_ylim()[1], 'r')
    ax.flatten()[2].plot(E0_x, E0_y/max(E0_y)*ax.flatten()[2].get_ylim()[1], 'r')
    ax.flatten()[3].plot(E0_x, E0_y/max(E0_y)*ax.flatten()[3].get_ylim()[1], 'r')
    fig = plt.gcf()
    fig.suptitle('All patient')
    plt.show()

    data = data[data['Chi_test']]
    data = data[data['J'] < 10000]
    data.reset_index(inplace=True)

    ax = data[['C50p', 'gamma', 'Emax', 'E0', 'J']].hist(bins=20)
    ax.flatten()[0].plot(C50p_x, C50p_y/max(C50p_y)*ax.flatten()[0].get_ylim()[1], 'r')
    ax.flatten()[1].plot(gamma_x, gamma_y/max(gamma_y)*ax.flatten()[1].get_ylim()[1], 'r')
    ax.flatten()[2].plot(E0_x, E0_y/max(E0_y)*ax.flatten()[2].get_ylim()[1], 'r')
    ax.flatten()[3].plot(E0_x, E0_y/max(E0_y)*ax.flatten()[3].get_ylim()[1], 'r')
    fig = plt.gcf()
    fig.suptitle('Signifiant patient')
    plt.show()

    plt.figure()
    Cep = np.linspace(0, 10)
    BIS = np.zeros((len(data), len(Cep)))
    for i in range(len(data)):
        for j in range(len(Cep)):
            Up = Cep[j]/data.loc[i, 'C50p']
            BIS[i, j] = data.loc[i, 'E0'] - data.loc[i, 'Emax'] * \
                Up ** data.loc[i, 'gamma'] / (1 + Up ** data.loc[i, 'gamma'])

        plt.plot(Cep, BIS[i, :], linewidth=0.5)

    plt.xlabel('Effect site Propofol (µg/ml)')
    plt.ylabel('BIS')
    plt.grid()
    plt.show()

    data = data[['Patient_id', 'Sex', 'Age', 'Height', 'Weight', 'C50p', 'gamma', 'Emax', 'E0']]
    data.to_csv('clean_table.csv')
