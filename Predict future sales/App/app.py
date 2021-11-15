import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pickle


def save_obj(obj_name, obj):
    st.session_state.update({obj_name: obj})

# functions
get_obj = st.session_state.get
delete_obj = st.session_state.pop
save_objs = st.session_state.update

def init_all_objs():
    items = pd.read_pickle('data/items.pickle', compression='bz2')
    item_categories = pd.read_csv('data/item_categories_1.csv')
    shops = pd.read_csv('data/shops_1.csv')
    shop_names = pd.read_csv('data/shops.csv')

    all_series = pd.read_pickle('data/all_series.pickle', compression='bz2')

    with open('data/lgbm_model.pickle', 'rb') as f:
        lgbm = pickle.load(f)

    save_objs({
        'lgbm': lgbm,
        'items': items,
        'item_categories': item_categories,
        'shops': shops,
        'shop_names': shop_names,
        'all_series': all_series
    })


if get_obj('start') is None:
    save_obj('start', True)
    init_all_objs()


items = get_obj('items')
item_categories = get_obj('item_categories')
shops = get_obj('shops')
shop_names = get_obj('shop_names')
lgbm = get_obj('lgbm')
all_series = get_obj('all_series')
X_cols = ['date_block_num', 'shop_id', 'item_id', 'item_category_id',
       'super_category', 'category', 'city', 'shop_type', 'shop_name',
       'item_cnt_lag1', 'item_cnt_lag2', 'item_cnt_lag3', 'item_cnt_lag4',
       'item_cnt_lag5', 'item_cnt_lag6', 'item_cnt_lag7', 'item_cnt_lag8',
       'item_cnt_lag9', 'item_cnt_lag10', 'item_cnt_lag11', 'item_cnt_lag12']

curr_month = 34
all_months = pd.DataFrame({'date_block_num': range(curr_month + 1)})
all_month_names = [datetime.datetime(2013 + i // 12, i % 12 + 1, 1).strftime('%B, %Y') for i in range(curr_month + 1)]

def predict(shop_id, item_id, month=curr_month):

    if shop_id in [0, 1, 10]:
        shop_id = {0: 57, 1: 58, 10: 11}[shop_id]

    X_test = pd.DataFrame({'shop_id':[shop_id], 'item_id': [item_id]})
    X_test = X_test.merge(items, on='item_id')\
            .merge(item_categories, on='item_category_id')\
            .merge(shops, on='shop_id')

    X_test['date_block_num'] = month

    history = all_series[(all_series.shop_id == shop_id) & (all_series.item_id == item_id)]
    if len(history) == 0:
        X_test[[f'item_cnt_lag{lag}' for lag in range(1, 13)]] = 0
    else:
        for lag in range(1, 13):
            X_test[f'item_cnt_lag{lag}'] = history[month - lag].iloc[0]

    X_test = X_test.fillna(0)[X_cols].drop(columns=['item_id', 'item_category_id', 'shop_id'])
    return np.clip(lgbm.predict(X_test)[0], 0, 20)

invalid_shop_items = False
cont = st.container()
with cont:
    st.text('Some values you can try to input:')
    st.markdown('''
|shop_id | item_id|
|--------|--------|
| 2 | 1905 |
| 59 | 20949 |
| 47 | 16787 |
''')
    cols = st.columns(2)
    with cols[0]:
        shop_id = st.number_input(label='Input shop id', format='%i', value=59)
        st.markdown('**The name of shop:**')
        try:
            st.text(shop_names[shop_names.shop_id == shop_id].iloc[0].shop_name)
        except:
            invalid_shop_items = True
            st.text('No such shop!')
    with cols[1]:
        item_id = st.number_input(label='Input item id', format='%i', value=22087)
        st.markdown('**The name of item:**')
        try:
            st.text(items[items.item_id == item_id].iloc[0].item_name)
        except:
            invalid_shop_items = True
            st.text('No such item!')

st.markdown('**Prediction**')
fig, ax = plt.subplots(figsize=(10,6))
history = all_series[(all_series.shop_id == shop_id) & (all_series.item_id == item_id)]
if not invalid_shop_items:
    predicted_curr = predict(shop_id, item_id)
    st.text(f'Predicted value: {predicted_curr}')
    predicted_prev = predict(shop_id, item_id, curr_month - 1)

st.markdown('**The plot of all sales**')
if len(history) == 0:
    st.text('there is no history')
else:
    ax.plot(history.T.iloc[2:])
    ax.plot([curr_month - 2, curr_month - 1,  curr_month],
            [history[curr_month - 2].iloc[0], predicted_prev, predicted_curr])
    ax.set_xticks(ticks=all_months.date_block_num)
    ax.set_xticklabels(all_month_names, rotation=80)
    ax.legend(['real sales over the all months', 'predicted values for current and previous months'])
    st.pyplot(fig)
    st.markdown(
f'''The absolute error of prediction for previous month is {round(abs(predicted_prev - history[curr_month - 1].iloc[0]), 5)}''')

# 2, 1905
# 59, 20949
# 47, 16787
