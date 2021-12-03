import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#####################################################
#
#                 Stat functions
#
#####################################################

# Exponential moving average: y_curr = (1-beta)*x_curr + beta*y_prev
def EMA(ser, beta):
    smoothed_ser = [ser[0]]
    for x in ser:
        smoothed_ser.append((1 - beta) * x + beta * smoothed_ser[-1])
    return smoothed_ser[1:]

# Inverse transforming to EMA: x_curr = (y_curr - beta * y_prev) / (1-beta)
def inv_EMA(smoothed_ser, beta):
    assert beta != 1, 'invalid value of beta for inversion'
    y = np.append([smoothed_ser[0]], smoothed_ser)
    return (y[1:] - beta * y[:-1]) / (1 - beta)

# Simple moving average. The first 'window' elements are cumulative average values.
def MA(ser, window):
    y = np.array(ser)
    smoothed_ser = [sum(ser[:k + 1]) / (k + 1) for k in range(window)]
    return np.append(smoothed_ser,
                     (sum(y[k:-(window - 1 - k)] for k in range(window - 1)) + y[window - 1:]) / window)

# Here the function for extracting the time series using days, weeks or months.
def time_series(df, store_id=None, time_range='day', agg_func='sum', all_values=None):
    time_col = f'ordered_{time_range}'
    if all_values is None:
        all_values = df[time_col].unique()
        all_values.sort()
    all_values = pd.DataFrame({time_col: all_values})
    mask = [True] * len(df) if store_id is None else df.Store == store_id
    return df.loc[mask, [time_col, 'target']].groupby(time_col).agg({'target': agg_func}).reset_index()\
        .merge(all_values, on=time_col, how='right').fillna(0).set_index(time_col).target

#####################################################
#
#                 Plot functions
#
#####################################################

# Make all borders invisible
def remove_axes():
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    
# The histogram of all values of categorical series using horizontal plot
def barh_series(ser, col_name='', title='', show_count=True):
    plt.figure(figsize=(10, 0.6 * len(ser)))
    try:
        labels = ser.index.astype('int')
        if (ser.index != labels).any():
            labels = ser.index
    except:
        labels = ser.index
    plt.barh(labels, width=ser.values, tick_label=labels)
    if show_count:
        for label in labels:
            if ser[label] < max(ser) / 30:
                plt.text(max(ser) / 30, label, ser[label], color='black', fontsize='large', ha='left', va='center')
            else:
                plt.text(ser[label] / 2, label, ser[label], color='white', fontsize='large', ha='center', va='center')

    plt.title(title)
    plt.yticks(labels, labels)
    if show_count:
        plt.xticks([])
    remove_axes()
    plt.ylabel(col_name)
    plt.show()
    
# The histogram of all values of categorical series in the column of a dataframe using horizontal plot
def barh(df, col_name, show_count=True):
    barh_series(df[col_name].value_counts().sort_index(),
                col_name,
                f'{col_name} histogram',
                show_count)

# The histogram of all values of categorical series using vertical plot
def bar_series(ser, col_name='', title='', show_count=True, figsize=(10, 6), rotation=0):
    plt.figure(figsize=figsize)
    try:
        labels = ser.index.astype('int')
        if (ser.index != labels).any():
            labels = ser.index
    except:
        labels = ser.index
    plt.bar(labels, height=ser.values)
    plt.xticks(ticks=labels, labels=labels, rotation=rotation)
    if show_count:
        for label in labels:
            if ser[label] < max(ser) / 30:
                plt.text(label, max(ser) / 30, ser[label], color='black', fontsize='large', ha='center', va='bottom')
            else:
                plt.text(label, ser[label] / 2, ser[label], color='white', fontsize='large', ha='center', va='center')

    plt.title(title)
    plt.xlabel(col_name)
    plt.ylabel('count')
    plt.show()
    
# The histogram of all values of categorical series in the column of a dataframe using vertical plot
def bar(df, col_name, title=None, show_count=True, figsize=(10, 6), rotation=0):
    bar_series(df[col_name].value_counts().sort_index(),
               col_name,
               f'{col_name} histogram' if title is None else title,
               show_count,
               figsize,
               rotation)
    
# The histogram of values of numeric series
def hist_series(ser, ser_name='', bins=60, figsize=(10, 6)):
    plt.figure(figsize=figsize)
    plt.hist(ser, bins=bins)
    plt.xlabel(ser_name)
    plt.ylabel('count')
    plt.title(f'{ser_name} histogram')
    plt.show()

# The histogram of values in the column of a dataframe
def hist(df, col_name, bins=60, figsize=(10, 6)):
    hist_series(df[col_name], col_name, bins, figsize)

# Cross plot of two features 
def cross_bar_plot(df, col_x, col_hue, figsize=(10, 6)):
    plt.figure(figsize=figsize)
    df_copy = df.copy()
    df_copy['Count'] = 0
    sns.barplot(data=df_copy.groupby([col_x, col_hue]).Count.count().reset_index(),
           x=col_x, y='Count', hue=col_hue
           )
    plt.title(f'cross plot of {col_x} and {col_hue}')
    plt.show()

def sales_title(time_range, agg_func):
    time_range_ly = time_range + 'ly' if time_range != 'day' else 'daily'
    sales_type = 'total' if agg_func == 'sum' else agg_func
    return f'{sales_type} {time_range_ly} sales '
    
# Plot the time series
def plot_time_series(df, store_id=None, time_range='month', agg_func='sum', all_values=None, title=None, figsize=(10, 6)):
    ser = time_series(df, store_id, time_range, agg_func, all_values)
    plt.figure(figsize=figsize)
    
    if title is None:
        title = sales_title(time_range, agg_func) + ('' if store_id is None else f'for store_id={store_id}') 
    plt.title(title)
    plt.xlabel(time_range)
    plt.ylabel('sales count')
    
    plt.plot(ser.index, ser)
    
# Plot the time series groupped by the category
def plot_time_series_of_cat(df, cat, time_range='month', agg_func='sum', all_values=None, weighted=True, title=None, figsize=(10, 6)):
    df_copy = pd.DataFrame({cat: df[cat]})
    df_copy[cat + 'count'] = 0
    cat_count = df_copy.groupby(cat).agg({cat + 'count': 'count'})
    cat_values = np.sort(cat_count.index)

    if all_values is None:
        all_values = df[f'ordered_{time_range}'].unique()
    
    if weighted:
        sers = [time_series(df[df[cat] == cat_value], time_range=time_range, agg_func=agg_func, all_values=all_values) / cat_count.loc[cat_value, cat + 'count']
            for cat_value in cat_values]
    else:
        sers = [time_series(df[df[cat] == cat_value], time_range=time_range, agg_func=agg_func, all_values=all_values)
            for cat_value in cat_values]
        
    plt.figure(figsize=figsize)
    
    if title is None:
        title = sales_title(time_range, agg_func) + f'for category {cat}' 
    plt.title(title)
    plt.xlabel(time_range)
    plt.ylabel('sales count')
    

    for ser in sers:
        plt.plot(ser.index, ser)
    plt.legend(cat_values)
    plt.show()

# Plot the time series groupped by the category that was encoded earlier using one-hot encoder
def plot_time_series_of_encoded_cat(df, cat, new_cats, old_values, time_range='month', agg_func='sum', all_values=None, weighted=True, title=None, figsize=(10, 6)):
    cat_target_df = pd.DataFrame(
        {cat: df[new_cats].idxmax(1).replace(to_replace=new_cats, value=old_values),
         'target': df.target,
        f'ordered_{time_range}': df[f'ordered_{time_range}']})
    plot_time_series_of_cat(cat_target_df, cat, time_range, agg_func, all_values, weighted, title, figsize)

# The same plot using the way of naming the encoded features
def my_plot_time_series_of_encoded_cat(df, cat, time_range='month', agg_func='sum', all_values=None, weighted=True, title=None, figsize=(10, 6)):
    new_cats = df.columns[df.columns.str.startswith(cat)]
    old_values = new_cats.str.replace(cat + '_', '')
    plot_time_series_of_encoded_cat(df, cat, new_cats, old_values, time_range, agg_func, all_values, weighted, title, figsize)