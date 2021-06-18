'''
This model type is common in Kaggle forecasting competitions.
Essentially, generate a bunch of statistical features for each time series and predict all 90 outputs at once.
'''

import pandas as pd
import numpy as np
import tensorflow as tf
import gc
from datetime import timedelta, date
from sklearn.preprocessing import LabelEncoder

NBAGS = 3 #Train and predict 3 times
score_ind = np.log(2) / 2
train = pd.read_csv("input/ventes_2018.csv",
                    parse_dates=['DATE'],
                    converters={'QTE': lambda u: np.log1p(float(u)) if float(u) > 0 else 0})

print("Filling all nans with 0")
print(train.shape)
train = pd.pivot_table(train, index=['ID_PDV','ID_ARTC'], columns=['DATE'], values=['QTE']).fillna(0)
train.columns = train.columns.get_level_values(1)
train.reset_index(inplace=True)
train = pd.melt(train, id_vars=['ID_PDV','ID_ARTC'], value_vars=train.columns[2:], value_name='QTE', var_name='DATE')
train['quarter'] = train['DATE'].dt.quarter
print(train.shape)

region = pd.read_csv("input/points_de_vente.csv")
products = pd.read_csv("input/nomenclature_produits.csv")
price = pd.read_csv("input/prix_vente.csv")
price['on_sale'] = 1

test_price = price[price.ANNEE == 2019].reset_index(drop=True)
test_price = pd.pivot_table(test_price, index=['ID_PDV','ID_ARTC'], columns=['TRIMESTRE'], values=['on_sale'])
test_price.columns = test_price.columns.get_level_values(1)
fillval = test_price.values
test_price = pd.DataFrame(index=test_price.index, columns = [c for c in pd.date_range(date(2019,1,1), periods=90, freq='D')])
for c in pd.date_range(date(2019,1,1), periods=90, freq='D'):
    test_price[c] = fillval

train = train.merge(price[price.ANNEE == 2018][['ID_PDV','ID_ARTC','TRIMESTRE','on_sale']], left_on=['ID_PDV','ID_ARTC','quarter'], right_on=['ID_PDV','ID_ARTC','TRIMESTRE'], how='left')

price = pd.pivot_table(train, index=['ID_PDV','ID_ARTC'], columns=['DATE'], values=['on_sale']).fillna(0)
price.columns = price.columns.get_level_values(1)

uids = train['ID_PDV'].astype(str) + '_' + train['ID_ARTC'].astype(str)
train = pd.pivot_table(train, index=['ID_PDV', 'ID_ARTC'], columns=['DATE'], values=['QTE']).fillna(0)
train.columns = train.columns.get_level_values(1)

scored = (train > 0).astype(int)
for i in range(1, 7):
    scored = scored + scored.shift(i, axis=1).fillna(0)
scored = (scored >= 1).astype(int) * score_ind

train = train + scored

item_group_mean = train.groupby('ID_ARTC').mean().reindex(train.index.get_level_values(1)) #Generate item level time series
store_group_mean = train.groupby('ID_PDV').mean().reindex(train.index.get_level_values(0)) #Generate store level time series

price = train.reset_index()[['ID_PDV','ID_ARTC']].merge(price.reset_index(), how='left', on=['ID_PDV','ID_ARTC'])
test_price = train.reset_index()[['ID_PDV','ID_ARTC']].merge(test_price.reset_index(), how='left', on=['ID_PDV','ID_ARTC']).fillna(0)

price = pd.concat([price, test_price.drop(['ID_PDV','ID_ARTC'], axis=1)], axis=1)
price.set_index(['ID_PDV','ID_ARTC'], inplace=True)
price.columns = [pd.to_datetime(c) for c in price.columns]

del test_price
gc.collect()

cat_features = train.reset_index()[['ID_PDV', 'ID_ARTC']]
cat_features = cat_features.merge(region, on='ID_PDV', how='left')
cat_features = cat_features.merge(products, on='ID_ARTC', how='left')

print(train.shape, price.shape, cat_features.shape)

for c in cat_features.columns:
    cat_features[c] = LabelEncoder().fit_transform(cat_features[c])

def get_timespan(df, dt, minus, periods, freq='D'):
    return df[pd.date_range(dt - timedelta(days=minus), periods=periods, freq=freq)]

def generate_features(df, price, t, is_train=True, name_prefix=None, use_price=False):
    X = {}
    for i in [3, 7, 14, 30, 60, 120]:
        tmp = get_timespan(df, t, i, i)
        X['diff_%s_mean' % i] = tmp.diff(axis=1).mean(axis=1).values
        X['mean_%s_decay' % i] = (tmp * np.power(0.9, np.arange(i)[::-1])).sum(axis=1).values
        X['mean_%s' % i] = tmp.mean(axis=1).values
        X['median_%s' % i] = tmp.median(axis=1).values
        X['min_%s' % i] = tmp.min(axis=1).values
        X['max_%s' % i] = tmp.max(axis=1).values
        X['std_%s' % i] = tmp.std(axis=1).values

    for i in range(1, 28):
        X['day_%s' % i] = get_timespan(df, t, i, 1).values.ravel()

    for i in range(7):
        X['mean_4_dow{}'.format(i)] = get_timespan(df, t, 28 - i, 4, freq='7D').mean(axis=1).values

    if(use_price):
        for i in range(-89, 90):
            X['on_sale_%s' % i] = get_timespan(price, t, i, 1).values.ravel()

    X = pd.DataFrame(X)
    if is_train:
        y = df[
            pd.date_range(t, periods=90)
        ].values
        return X, y
    if name_prefix is not None:
        X.columns = ['%s_%s' % (name_prefix, c) for c in X.columns]
    return X


'''
For 12 weeks, generate a series of item, store, and item-store level features, 
append the categorical information and stack all days into 1 dataset
'''
num_days = 12
t = date(2019, 1, 1) - timedelta(num_days * 7 + 91 +7)
t + timedelta(num_days*7 + 90)
X_l, y_l = [], []
for i in range(num_days):
    delta = timedelta(days=7 * i)
    print(f"Generating Features for {t + delta} with weekday: {(t + delta).weekday()}")
    X_tmp, y_tmp = generate_features(train, price, t + delta, use_price=True)
    X_tmp2 = generate_features(item_group_mean, price, t + delta, use_price=False, is_train=False, name_prefix='item')
    X_tmp3 = generate_features(store_group_mean, price, t + delta, use_price=False, is_train=False, name_prefix='store')
    X_tmp = pd.concat([X_tmp, X_tmp2, X_tmp3], axis=1)
    X_l.append(X_tmp)
    y_l.append(y_tmp)
    del X_tmp, X_tmp2, X_tmp3

X_train = pd.concat(X_l, axis=0)
y_train = np.concatenate(y_l, axis=0)

print(X_train.shape, y_train.shape)
del X_l, y_l
gc.collect()


'''
Similar to the cnn, I shift my test set by two weeks to avoid the holidays.
I would not do this if the end of the time series (arguably the most important part) was so noisy.
'''
X_test = generate_features(train.shift(14, axis=1), price, date(2019, 1, 1), is_train=False, use_price=True)
X_test2 = generate_features(item_group_mean.shift(14, axis=1), price, date(2019, 1, 1), is_train=False, use_price=False, name_prefix='item')
X_test3 = generate_features(store_group_mean.shift(14, axis=1), price, date(2019, 1, 1), is_train=False, use_price=False, name_prefix='store')
X_test = pd.concat([X_test, X_test2, X_test3], axis=1)

del X_test2, X_test3
gc.collect()

print(X_train.shape, X_test.shape)
X_train.fillna(0, inplace=True)
X_test.fillna(0, inplace=True)

# RMSLE which ignores "0" observations. Since we incremented our training set by a score indicator, all scored values will be non-zeros
def custom_loss(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true[y_true != 0] - y_pred[y_true != 0])))

def get_model():
    inp = tf.keras.layers.Input(shape=(X_train.shape[1],))

    x = tf.keras.layers.Dense(256)(inp)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(.1)(x)

    x = tf.keras.layers.Dense(256)(x)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(.1)(x)

    x = tf.keras.layers.Dense(128)(x)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(.1)(x)

    x = tf.keras.layers.Dense(128)(x)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(.05)(x)

    output = tf.keras.layers.Dense(90, activation='relu')(x)

    model = tf.keras.models.Model(inputs=inp, outputs=[output])
    model.compile(optimizer='adam', loss=custom_loss)
    return model

def scheduler(epoch, lr):
    if epoch < 12:
        return lr
    elif epoch < 16:
        return lr/10
    else:
        return lr/100

test_pred = np.zeros((len(train), 90))
for bag in range(NBAGS):
    print("Training model with holdout data")
    callback = [tf.keras.callbacks.LearningRateScheduler(scheduler)]

    model = get_model()
    model.fit(X_train, y_train, epochs=20, batch_size=2048, verbose=1, callbacks=callback, validation_split=0.05) #validation split isn't really necessary

    test_pred += model.predict(X_test) / NBAGS

del X_train, X_test
gc.collect()

test_pred[:,0] = 0 #Set 1/1 to 0
test = train.reset_index()[['ID_PDV','ID_ARTC']]
test_pred = pd.DataFrame(test_pred, columns=pd.date_range(date(2019,1,1), periods=90, freq='D'))
test = pd.concat([test, test_pred], axis=1)
sub = pd.melt(test, id_vars=['ID_PDV','ID_ARTC'], value_vars=pd.date_range(date(2019,1,1), periods=90, freq='D'),
              value_name='QTE', var_name='DATE')
sub['datestr'] = sub['DATE'].astype(str)
sub['id'] = sub['ID_PDV'].astype(str) + '_' + sub['ID_ARTC'].astype(str) + '_' + sub['datestr'].str.replace('-','')
sub_raw = sub.copy()
sub_raw['QTE'] = np.clip(np.expm1(sub['QTE'] - score_ind), 0, 100000)
sub['QTE'] = np.clip(np.round(np.expm1(sub['QTE'] - score_ind)), 0, 100000).astype(int)


sub = sub[['id','QTE']]
sub_raw = sub_raw[['id','QTE']]
sub['QTE'] = sub['QTE'].replace(0, np.nan)
sub = sub[sub.QTE.notnull()].reset_index(drop=True)
sub.to_csv("output/submission_mlp.csv", index=False) #Scores 0.563

sub_raw.to_csv("raw_predictions/mlp.csv", index=False, header=True) # Save raw predictions for blending
