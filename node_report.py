# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import psycopg2
import pandas as pd
import pandas.io.sql as sqlio
import matplotlib.pyplot as plt
import numpy as np
import datetime
import time
import collections
from pandas._libs.lib import is_integer
from scipy.stats import pearsonr, spearmanr, f_oneway, variation
from scipy.stats import linregress

# stepstone VPN
new_host = "44.227.86.163"
conn = psycopg2.connect(host=new_host, port='6432', dbname="clickcast", user="clickcast_ba_ro", 
                      password='R1tiKGHHAty3x5k')


def connect():
    global conn
    conn = psycopg2.connect(host=new_host, port='6432', dbname="clickcast", user="clickcast_ba_ro", 
                      password='R1tiKGHHAty3x5k')

def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def calculate_rolling_aggs(df, col):
    grouped = df.groupby(col).agg({'clicks':'sum','spend':'sum', 'clicks': 'sum',
                                  'paid':'sum'})
    grouped['cpc'] = grouped.spend / grouped.clicks
    grouped['cpa'] = grouped.spend / grouped.paid
    grouped['cr'] = grouped.paid / grouped.clicks
    
    grouped['rolling_clicks'] = grouped.clicks.cumsum()
    grouped['rolling_applies'] = grouped.paid.cumsum()
    
    df_out = grouped
    
    return df_out.sort_values('clicks', ascending=False)

def cpa_growth(df, show=False, bins=10):
    df['cpc_quantile'] = weighted_qcut(df.cpc, weights=df.clicks, q=bins, labels=range(bins))
    dfg = calculate_rolling_aggs(df, 'cpc_quantile').sort_index()
    max_cpa = dfg.cpa.max()
    xdata = np.linspace(0, max_cpa, bins)
    if show:
        plt.plot(xdata, dfg.rolling_applies.values)
    return dfg   

def cpa_bin(df, bins=5):
    df['cpc_quantile'] = weighted_qcut(df.cpa, weights=df.clicks, q=bins, labels=range(bins))
    dfg = calculate_rolling_aggs(df, 'cpc_quantile').sort_index()
    return dfg

def weighted_qcut(values, weights, q, **kwargs):
    'Return weighted quantile cuts from a given series, values.'
    if is_integer(q):
        quantiles = np.linspace(0, 1, q + 1)
    else:
        quantiles = q
    order = weights.iloc[values.argsort()].cumsum()
    bins = pd.cut(order / order.iloc[-1], quantiles, **kwargs)
    return bins.sort_index()

def node_stats(node, start_date, end_date):
    connect()
    ma_window = 3
    sql = """
    select  date(jsm.created_at) as created_date,
        coalesce(sum(jsm.price) filter (where jsm.global_action < 10 and jsm.paid=true 
                and jsm.organic=false and jsm.job_board_id in (22164, 22453, 22483, 22172)), 0) as spend_indeed,
        coalesce(sum(jsm.price) filter(where paid and not jsm.organic and jb.enterprise_id = 114
                                         and bid_metric_matches(jsm.bid_metric, global_action)
                                      ), 0) as spend_acx,
        count(*) filter(where jsm.global_action = 12 and paid=true and jb.enterprise_id <> 114) as paid_indeed,
        count(*) filter(where jsm.global_action = 12 and paid=true and jb.enterprise_id = 114) as paid_acx
    FROM job_stats_master jsm
            join job_boards jb on jsm.job_board_id = jb.id
    join job_groups jg on jsm.job_group_id = jg.id
    join labor_orders lo on jg.labor_order_id = lo.id
    where jsm.employer_id in (8984, 8992)
    and lo.node = '{}'
      and jsm.created_at between '{}' and '{}'
      group by created_date
      order by created_date""".format(node, start_date, end_date)
    #print(sql)
    df = sqlio.read_sql_query(sql, conn)
    if len(df) < 7:
        return 0
    # print('DB done')
    metrics = ['spend', 'paid']
    nets = ['acx', 'indeed', 'total']
    for metric in metrics:
        df[metric + '_total'] = df[metric + '_acx'] + df[metric + '_indeed']
    
    for net in nets:
        df['cpa_' + net] = df['spend_' + net] / df['paid_' + net]
        for metric in metrics:
            df[metric + '_' + net + '_ma'] = df[metric + '_' + net].rolling(window=ma_window).mean()
        df['cpa_' + net + '_ma'] = df['spend_' + net + '_ma'] / df['paid_' + net + '_ma']
        

    
    spearman_co = {}
    diff_to_max = {}
    for net in nets:
        diff_to_max[net] = {}
        spearman_co[net] = {}
        for ma in ['', '_ma']:
            spearman_co[net][ma] = {}
            t = spearmanr(df.dropna()['cpa_' + net + ma], df.dropna()['paid_' + net + ma])
            spearman_co[net][ma]['corr'] = t[0]
            spearman_co[net][ma]['p'] = t[1] 
            
            for m in metrics + ['cpa']:
                m_max = min(df[m + '_' + net + ma].max(), 10000)
                m_last = df[m + '_' + net + ma].iloc[-1]
                try:
                    diff_to_max[net][m + ma] = (m_max - m_last) / m_max
                except:
                    print("net={}, ma={}, metric={}".format(net, ma, m))
                    print("m_max={}, m_last={}".format(m_max, m_last))
    # last week stats
    week_ago = pd.Timestamp('2021-03-05') - pd.Timedelta('7 days') 
    last_week_applies = df[df['created_date']>=week_ago]['paid_total'].sum()
    last_week_cpa = df[df['created_date']>=week_ago]['spend_total'].sum() / last_week_applies
    
    # do lin regress
    x = list(range(12))
    try:
        y = df['paid_total'].iloc[-12:]
        apply_slope, intercept, r, apply_p, se = linregress(x, y)
    except:
        apply_slope, intercept, r, apply_p, se = [0]*5
        
    try:
        y = df['cpa_total'].iloc[-12:]
        cpa_slope, intercept, r, cpa_p, se = linregress(x, y)
    except:
        cpa_slope, intercept, r, cpa_p, se = [0]*5


    return {'last_week_applies': last_week_applies, 'last_week_cpa': last_week_cpa, 
            'data': df, 'spearman_co': spearman_co, 'diff_to_max': diff_to_max, 'apply_slope': apply_slope, 
            'apply_p': apply_p, 'cpa_slope': cpa_slope, 'cpa_p': cpa_p}

    

def predict_volume(cpa_now, apply_now, cpa_diff, apply_diff, total_corr=0, apply_slope=0, cpa_slope=0):
    if cpa_diff < 0.31:
        if apply_diff < 0.81:
            # cpa high, applies high
            coefs_cpa_plus = [1.06, 1.1, 1.15]
            coefs_cpa_minus = [0.1, 0.3, 0.6]
            coef_market = [0.7, 1, 1.1]
        else:
            # cpa high, applies low
            coefs_cpa_plus = [1.02, 1.04, 1.04]
            coefs_cpa_minus = [0.05, 0.2, 0.3]
            coef_market = [0.5, 1, 1.05]
    else:
        if apply_diff < 0.81:
            # cpa low, applies high
            # coefs_cpa_plus = [1.2, 1.35, 1.45]
            coefs_cpa_plus = [1.25, 1.45, 1.60]
            coefs_cpa_minus = [0.1, 0.3, 0.5]
            coef_market = [0.7, 1, 1.1]
        else:
            # cpa low, applies low
            coefs_cpa_plus = [1.06, 1.1, 1.15]
            coefs_cpa_minus = [0.1, 0.3, 0.6]
            coef_market = [0.7, 1, 1.1]   
    res = np.zeros((7, 3))
    for i, cm in enumerate(coef_market):
        for j, cp in enumerate(coefs_cpa_minus):
            res[j, i] = coef_market[i]*coefs_cpa_minus[j]
        res[3, i] = coef_market[i]
        for j, cp in enumerate(coefs_cpa_minus):
            res[j+4, i] = coef_market[i]*coefs_cpa_plus[j]
    res = res*apply_now
    cpa_column = [c*cpa_now for c in [0.25, 0.5, 0.75, 1, 1.25, 1.50, 1.75]]
    res = np.insert(res, 0, values=cpa_column, axis=1)
    return res


def nodes_data(start_date, end_date, nodes):
    
    df = pd.DataFrame([])
    for n in nodes:
        #print(n)
        res = node_stats(n, start_date, end_date)
        if res == 0:
            continue
        d = dict(flatten(res['spearman_co']), **flatten(res['diff_to_max']))
        d.update()
        row = pd.DataFrame(d, index=[n])
        for k in ['apply_slope', 'apply_p', 'cpa_slope', 'cpa_p', 'last_week_applies', 'last_week_cpa']:
            row[k] = res[k]
            """
        row['apply_slope'] = res['apply_slope']
        row['apply_p'] = res['apply_p']
        row['cpa_slope'] = res['cpa_slope']
        row['cpa_p'] = res['cpa_p']
            """
        df = df.append(row)    
    return df

def custom_round(x, base=5):
    return int(base * round(float(x)/base))


def produce_report(end_date):
    nodes = get_nodes(end_date)
    start_date = (pd.Timestamp(end_date) - pd.Timedelta('7 days')).strftime('%Y-%m-%d')
    df_nodes = nodes_data(start_date, end_date, nodes=nodes)
    #print("df_nodes.index{}".format(df_nodes.index))
    df_res = pd.DataFrame([])
    for n in df_nodes.index:
        print(n)
        row = df_nodes.loc[n, :]
        #print("row:{}".format(row))
        pred = predict_volume(cpa_now=row['last_week_cpa'], apply_now=row['last_week_applies'], 
                              cpa_diff=row['total_cpa'], apply_diff=row['total_paid'])
        new_df = pd.DataFrame(pred)
        new_df['Node'] = n
        df_res = df_res.append(new_df)
        #df_res = df_res.append(pd.Series(), ignore_index=True)#.fillna(' ')
    

    df_res['CPA'] = df_res.loc[:, 0].round()    
    df_res['Applies_Easier'] = df_res.loc[:, 3].apply(lambda x: custom_round(x))
    df_res['Budget_Easier'] = (df_res.loc[:, 'CPA']*df_res.loc[:, 'Applies_Easier']).round()
    df_res['Applies_Current'] = df_res.loc[:, 2].apply(lambda x: custom_round(x))
    df_res['Budget_Current'] = (df_res.loc[:, 'CPA']*df_res.loc[:, 'Applies_Current']).round()
    df_res['Applies_Harder'] = df_res.loc[:, 1].apply(lambda x: custom_round(x))
    df_res['Budget_Harder'] = (df_res.loc[:, 'CPA']*df_res.loc[:, 'Applies_Harder']).round()
    df_res.fillna(' ')
    
    file_name = 'amz_estimate_' + end_date + '.csv'
    df_res.to_csv(file_name)
    print("Saved {}".format(file_name))
    return df_res[['Node', 'CPA', 'Applies_Easier', 'Budget_Easier', 'Applies_Current', 'Budget_Current',
                  'Applies_Harder', 'Budget_Harder']] 



def get_nodes(report_date, apply_min=50):
    connect()
    start_date = (pd.Timestamp(report_date) - pd.Timedelta('7 days')).strftime('%Y-%m-%d')
    sql = """
    select  lo.node,
            count(*) filter(where jsm.global_action = 12 and paid=true) as paid
    FROM job_stats_master jsm
            join job_boards jb on jsm.job_board_id = jb.id
            join job_groups jg on jsm.job_group_id = jg.id
            join labor_orders lo on jg.labor_order_id = lo.id
    where jsm.employer_id in (8984, 8992)
      and jsm.created_at between '{}' and '{}'
      group by lo.node
      having count(*) filter(where jsm.global_action = 12 and paid=true) > {}
      order by lo.node""".format(start_date, report_date, apply_min)
    df = sqlio.read_sql_query(sql, conn)
    return list(df['node'])
