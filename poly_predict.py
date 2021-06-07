import sys
sys.path.insert(0, '..')

from utils import data
import os
import sklearn
import numpy as np
import json
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')


# ------------ HYPERPARAMETERS -------------
BASE_PATH = '../COVID-19/csse_covid_19_data/'
MIN_CASES = 5000
# ------------------------------------------

confirmed = os.path.join(
    BASE_PATH,
    'csse_covid_19_time_series',
    'time_series_19-covid-Confirmed.csv')
confirmed = data.load_csv_data(confirmed)
features = []
targets = []





fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111)
cm = plt.get_cmap('jet')
NUM_COLORS = 0
LINE_STYLES = ['solid', 'dashed', 'dotted']
NUM_STYLES = len(LINE_STYLES)


for val in np.unique(confirmed["Country/Region"]):
    df = data.filter_by_attribute(
        confirmed, "Country/Region", val)
    cases, labels = data.get_cases_chronologically(df)
    cases = cases.sum(axis=0)

    if cases.sum() > MIN_CASES:
        NUM_COLORS += 1

colors = [cm(i) for i in np.linspace(0, 1, NUM_COLORS)]
legend = []
handles = []



predictions = {}
degrees = np.arange(10)
for val in np.unique(confirmed["Country/Region"]):
    df = data.filter_by_attribute(
        confirmed, "Country/Region", val)
    cases, labels = data.get_cases_chronologically(df)
    cases = cases.sum(axis=0)
    pred_week, pred_month, pred_year = [], [], []
    countries = np.sum(np.array(cases, dtype='float'), axis=0)
    time = np.arange(countries.shape[1])
    next_week = np.arange(time.shape[0], time.shape[0]+7)
    next_month = np.arange(time.shape[0], time.shape[0] + 31)
    next_year = np.arange(time.shape[0], time.shape[0] + 365)
    for j in degrees:
        z = np.polyfit(time, countries, j)
        p = np.poly1d(z)
        pred_week.append(p(next_week))
        pred_month.append(p(next_month))
        pred_year.append(p(next_year))
    if val not in predictions:
        predictions[val] = {}
    predictions[val]['week'] = pred_week
    predictions[val]['month'] = pred_month
    predictions[val]['year'] = pred_year

    if cases.sum() > MIN_CASES:
        i = len(legend)
        lines = ax.plot(predictions[val]['week'][3], label=labels[0,1])
        handles.append(lines[0])
        lines[0].set_linestyle(LINE_STYLES[i%NUM_STYLES])
        lines[0].set_color(colors[i])
        legend.append(labels[0, 1])


ax.set_ylabel('# of confirmed cases')
ax.set_xlabel("Time (days since Jan 22, 2020)")

ax.set_yscale('log')
ax.legend(handles, legend, bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4)
plt.tight_layout()
plt.savefig('results/poly_week_predict.png')
