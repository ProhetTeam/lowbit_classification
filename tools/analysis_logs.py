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
        'weight_quant_error_loss',
        'weight_quant_std_loss',
        'weight_quant_cover_ratio_loss',
        'quant_std_loss']

use_one_fig = True
log_path = 'work_dirs/ABQAT/config1_res18_abqat_int4_lr4x_4m/20210427_181319.log.json'

def load_json_logs(json_log):
    log_dicts = defaultdict(list)
    with open(json_log, 'r') as log_file:
        for line in log_file:
            log = json.loads(line.strip())
            if 'epoch' not in log:
                    continue
            for key in keys:
                try:
                    log_dicts[key].append(log[key])
                except KeyError:
                    print(f'{key} is NOT exists!')
    return log_dicts

log_dicts = load_json_logs(log_path)
if use_one_fig:
    fig = go.Figure()
else:
    subplot_titles = tuple(keys)
    specs = [[{"type": "Scatter"}]] * len(keys)
    fig = make_subplots(
        rows = len(keys), cols = 1,
        specs = specs,
        subplot_titles= subplot_titles)
idx = 1
for k, v in log_dicts.items():
    if use_one_fig:
        fig.add_trace(go.Scatter(x= np.arange(0, len(v)), y = np.array(v),
                            mode='lines',
                            name= k))
    else:
        fig.add_trace(go.Scatter(x= np.arange(0, len(v)), y = np.array(v),
                            mode='lines',
                            name= k), row = idx, col = 1)
    idx += 1
fig.write_html('log.html')
print("DONE!!")