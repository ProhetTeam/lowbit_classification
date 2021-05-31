import argparse
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import plotly.express as px
import plotly.offline as of
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from collections import defaultdict
import plotly.figure_factory as ff
import pandas as pd

of.offline.init_notebook_mode(connected=True)

keys = ['top-1',
        'top-5']

use_one_fig = False
logs_path = ['work_dirs/LSQDPlus/config1_res18_lsq_int4_updatelr4x_4m/20210518_113809.log.json',
              'work_dirs/LSQDPlus/config2_res18_lsqdplus_int4_updatelr4x_offset_4m/20210519_095439.log.json',
              'work_dirs/LSQDPlus/config3_res18_lsqdplus_int4_updatelr4x_init3_4m/20210518_230312.log.json',
              "work_dirs/LSQDPlus/config4_res18_lsqdplus_int4_updatelr4x_weightloss_4m/20210519_100857.log.json",
              "work_dirs/LSQDPlus/config4_res18_lsqdplus_int4_lr4x_convQuantloss_4m/20210519_185754.log.json"]
extra_name = ["res18_lsq_int4",
              "res18_lsqdplus_offset",
              "res18_lsqdplus_init3",
              "res18_lsqdplus_weightlayerloss",
              "res18_lsqdplus_actlayerloss"]

def load_json_logs(json_log, extra_name = ""):
    log_dicts = defaultdict(list)
    with open(json_log, 'r') as log_file:
        for line in log_file:
            log = json.loads(line.strip())
            if 'epoch' not in log or log['mode'] != 'val':
                    continue
            for key in keys:
                name = key if extra_name == "" else extra_name + "_" + key 
                try:
                    log_dicts[name].append(log[key])
                except KeyError:
                    print(f'{key} is NOT exists!')
    return log_dicts


log_dicts_sum = defaultdict(list)
for log_id, log_path in enumerate(logs_path):
    log_dicts = load_json_logs(log_path,  extra_name[log_id])
    log_dicts_sum.update(log_dicts)

if use_one_fig:
    fig = go.Figure()
else:
    subplot_titles = tuple(keys)
    specs = [[{"type": "Scatter"}]] * len(keys)
    fig = make_subplots(
        rows = len(keys), cols = 1,
        specs = specs,
        subplot_titles= subplot_titles)
idx = 0
for k, v in log_dicts_sum.items():
    if use_one_fig:
        fig.add_trace(go.Scatter(x= np.arange(0, len(v)), y = np.array(v),
                            mode='lines',
                            name= k))
    else:
        fig.add_trace(go.Scatter(x= np.arange(0, len(v)), y = np.array(v),
                            mode='lines',
                            name= k), row = idx % len(keys) + 1, col = 1)
    idx += 1
fig.write_html('log.html')
print("DONE!!")