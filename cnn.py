'''
The modeling approach was inspired by Lenz Du's awesome solution to the corporacion favorita kaggle competition.
If you find this script helpful, please check out his github and direct credit to him.
https://github.com/LenzDu/Kaggle-Competition-Favorita
'''
import pandas as pd
import numpy as np

import tensorflow as tf
import gc

from datetime import timedelta, date
from sklearn.preprocessing import LabelEncoder

score_ind = np.log(2)/2 #This is a value used to indicate whether a value is scored or not
NBAGS = 3 #Number of bagged runs
TIMESTEPS = 120 #Number of lags to include
train = pd.read_csv("input/ventes_2018.csv",
                    parse_dates=['DATE'],
                    converters={'QTE': lambda u: np.log1p(float(u)) if float(u) > 0 else 0})


#Very important to fill all missing rows with 0s before merging with price data to avoid a data leak
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

#Fill every date in the test time window with the price
#I ended up using just on_sale to indicate if it could be sold
test_price = price[price.ANNEE == 2019].reset_index(drop=True)
test_price = pd.pivot_table(test_price, index=['ID_PDV','ID_ARTC'], columns=['TRIMESTRE'], values=['on_sale'])
test_price.columns = test_price.columns.get_level_values(1)
fillval = test_price.values
test_price = pd.DataFrame(index=test_price.index, columns = [c for c in pd.date_range(date(2019,1,1), periods=90, freq='D')])
for c in pd.date_range(date(2019,1,1), periods=90, freq='D'):
    test_price[c] = fillval


#Merge price data
train = train.merge(price[price.ANNEE == 2018][['ID_PDV','ID_ARTC','TRIMESTRE','on_sale']], left_on=['ID_PDV','ID_ARTC','quarter'], right_on=['ID_PDV','ID_ARTC','TRIMESTRE'], how='left')

price = pd.pivot_table(train, index=['ID_PDV','ID_ARTC'], columns=['DATE'], values=['on_sale']).fillna(0)
price.columns = price.columns.get_level_values(1)

train = pd.pivot_table(train, index=['ID_PDV','ID_ARTC'], columns=['DATE'], values=['QTE']).fillna(0)
train.columns = train.columns.get_level_values(1)
print(train.shape, price.shape, test_price.shape)

#Generating categorical features
cat_features = train.reset_index()[['ID_PDV']].merge(region, how='left', on='ID_PDV').drop(['ID_PDV'], axis=1)
cat_features2 = train.reset_index()[['ID_ARTC']].merge(products, how='left', on='ID_ARTC').drop(['ID_ARTC'], axis=1)
cat_features = pd.concat([cat_features, cat_features2], axis=1)
for c in cat_features.columns:
    print(c, len(cat_features[c].unique()))

price = train.reset_index()[['ID_PDV','ID_ARTC']].merge(price.reset_index(), how='left', on=['ID_PDV','ID_ARTC'])
test_price = train.reset_index()[['ID_PDV','ID_ARTC']].merge(test_price.reset_index(), how='left', on=['ID_PDV','ID_ARTC']).fillna(0)

price = pd.concat([price, test_price.drop(['ID_PDV','ID_ARTC'], axis=1)], axis=1)
price.set_index(['ID_PDV','ID_ARTC'], inplace=True)
price.columns = [pd.to_datetime(c) for c in price.columns]

del test_price
gc.collect()

#if not onsale, fill with weekday mean, this didn't really add anything to my model
train2 = train.copy()
train2[price == 0] = np.nan
for i in range(7):
    cols = [c for c in train2.columns if c.weekday() == i]

    for j, c in enumerate(cols):
        fillvals = train2[cols[:j]].mean(axis=1).values
        train2[c][np.isnan(train2[c])] = fillvals[np.isnan(train2[c])]

train2.fillna(0, inplace=True)

'''
One trick that helped a lot was identifying which records were scored.
The metric is a custom RMSLE value, in which observations with no sales in 
the last 7 days were not scored. I used the code below to identify scored records
and then add an indicator to my training data. For this script, I use a custom 
loss function to ignore non-scored items
'''
## Identifying scored records and incrementing by 1 in prep of custom rmse loss
scored = (train > 0).astype(int)
for i in range(1,7):
    scored = scored + scored.shift(i, axis=1).fillna(0)
scored = (scored >= 1).astype(int) * score_ind
train = train + scored

# Function to pull in a sequence ending the day before the prediction date.
def get_timespan(df,  pred_start, timesteps=TIMESTEPS, is_train=True):
    X = df[pd.date_range(pred_start-timedelta(days=timesteps), pred_start-timedelta(days=1))].values
    if is_train:
        y = df[pd.date_range(pred_start, periods=90)].values
    else:
        y = None
    return X, y

# Generator to feed sequences to the model
def train_generator(df, df2, price, cat_features, first_pred_start, batch_size=2000, n_range=20, day_skip=1):
    encoder = LabelEncoder()
    voct = encoder.fit_transform(cat_features['ID_VOCT'])
    regn = encoder.fit_transform(cat_features['ID_REGN'])
    cais_grp = encoder.fit_transform(cat_features['NB_CAIS_GRP'])
    surf_grp = encoder.fit_transform(cat_features['SURF_GRP'])
    rayn = encoder.fit_transform(cat_features['LB_VENT_RAYN'])
    faml = encoder.fit_transform(cat_features['LB_VENT_FAML'])

    item_group_mean = df.groupby('ID_ARTC').mean().reindex(df.index.get_level_values(1))
    store_group_mean = df.groupby('ID_PDV').mean().reindex(df.index.get_level_values(0))
    cat_features = np.stack([voct, regn, cais_grp, surf_grp, rayn, faml], axis=1)
    while 1:
        date_part = np.random.permutation(range(n_range))
        for i in date_part:
            keep_idx = np.random.permutation(df.shape[0])[:batch_size]
            df_tmp = df.iloc[keep_idx,:]
            df2_tmp = df2.iloc[keep_idx, :]
            price_tmp = price.iloc[keep_idx, :]
            item_tmp = item_group_mean.iloc[keep_idx,:]
            store_tmp = store_group_mean.iloc[keep_idx,:]
            cat_tmp = cat_features[keep_idx]

            pred_start = first_pred_start + timedelta(days=int(day_skip*i))
            yield create_dataset_part(df_tmp, df2_tmp, price_tmp, item_tmp, store_tmp, cat_tmp, pred_start, True)
            gc.collect()

def create_dataset(df, df2, price, cat_features, pred_start, is_train=True):
    encoder = LabelEncoder()
    voct = encoder.fit_transform(cat_features['ID_VOCT'])
    regn = encoder.fit_transform(cat_features['ID_REGN'])
    cais_grp = encoder.fit_transform(cat_features['NB_CAIS_GRP'])
    surf_grp = encoder.fit_transform(cat_features['SURF_GRP'])
    rayn = encoder.fit_transform(cat_features['LB_VENT_RAYN'])
    faml = encoder.fit_transform(cat_features['LB_VENT_FAML'])


    item_group_mean = df.groupby('ID_ARTC').mean().reindex(df.index.get_level_values(1))
    store_group_mean = df.groupby('ID_PDV').mean().reindex(df.index.get_level_values(0))

    cat_features = np.stack([voct, regn, cais_grp, surf_grp, rayn, faml], axis=1)
    return create_dataset_part(df, df2, price, item_group_mean, store_group_mean, cat_features, pred_start, is_train)

def create_dataset_part(df, df2, price, item_group_mean, store_group_mean, cat_features, pred_start, is_train=True):
    x, y = get_timespan(df, pred_start,timesteps=TIMESTEPS, is_train=is_train)
    x2, _ = get_timespan(df2, pred_start, timesteps=TIMESTEPS, is_train=False)
    is0 = ((x - (np.log(2)/2)) <=0).astype(int)
    x_price, _ = get_timespan(price, pred_start+timedelta(90), timesteps=TIMESTEPS+90, is_train=False)

    x_item, _ = get_timespan(item_group_mean, pred_start, is_train=False)
    x_store, _ = get_timespan(store_group_mean, pred_start, is_train=False)

    weekday = np.tile([d.weekday() for d in pd.date_range(pred_start - timedelta(days=TIMESTEPS), periods=TIMESTEPS + 90)],(x.shape[0], 1))

    x = x.reshape((-1, TIMESTEPS, 1))
    x2 = x2.reshape((-1, TIMESTEPS, 1))
    is0 = is0.reshape((-1, TIMESTEPS, 1))
    x_price = x_price.reshape((-1, TIMESTEPS+90, 1))
    x_item = x_item.reshape((-1, TIMESTEPS, 1))
    x_store = x_store.reshape((-1, TIMESTEPS, 1))

    return([x, x2, is0, x_price, x_item, x_store, weekday, cat_features], y)


n_range = 18 #Choose from 18 weeks to start from
train_pred_start = date(2019,1,1) - timedelta(91) - timedelta(7*n_range)
print(f"Training sequences start {train_pred_start - timedelta(TIMESTEPS)}")
print(f"Training predictions start {train_pred_start}")
print(f"Training predictions end {train_pred_start + timedelta(7*n_range + 90)}")
train_set = train_generator(train,
                            train2,
                            price,
                            cat_features,
                            first_pred_start=train_pred_start,
                            day_skip=7,
                            n_range=n_range)


'''
One of the toughest parts of this dataset is the fact that the last week of 2018 is surely an outlier with Christmas, 
Christmas Eve, and New Year's Eve. To account for this, I actually generated my testing sequences as if they ended 2 weeks 
earlier. This made my score a little more consistent, but is not something I would use during other times of the year.
'''
Xtest, _ = create_dataset(train.shift(14, axis=1), train2.shift(14, axis=1), price, cat_features, date(2019,1,1), is_train=False)

# RMSLE which ignores "0" observations. Since we incremented our training set by a score indicator, all scored values will be non-zeros
def custom_loss(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true[y_true != 0] - y_pred[y_true != 0])))

def get_model():
    latent_dim=100
    seq_in = tf.keras.layers.Input(shape=(TIMESTEPS, 1))
    week_seq_in = tf.keras.layers.Input(shape=(TIMESTEPS, 1))
    is0_in = tf.keras.layers.Input(shape=(TIMESTEPS, 1))
    price_in = tf.keras.layers.Input(shape=(TIMESTEPS + 90, 1))
    weekday_in = tf.keras.layers.Input(shape=(TIMESTEPS + 90,), dtype='uint8')
    item_mean_in = tf.keras.layers.Input(shape=(TIMESTEPS, 1))
    store_mean_in = tf.keras.layers.Input(shape=(TIMESTEPS, 1))
    cat_features = tf.keras.layers.Input(shape=(6,))

    weekday_embed_encode = tf.keras.layers.Embedding(7, 4, input_length=TIMESTEPS + 90)(weekday_in)

    voct = tf.keras.layers.Lambda(lambda x: x[:, 0, None])(cat_features)
    regn = tf.keras.layers.Lambda(lambda x: x[:, 1, None])(cat_features)
    cais_grp = tf.keras.layers.Lambda(lambda x: x[:, 2, None])(cat_features)
    surf_grp = tf.keras.layers.Lambda(lambda x: x[:, 3, None])(cat_features)

    voct_embed = tf.keras.layers.Flatten()(tf.keras.layers.Embedding(5, 2, input_length=1)(voct))
    regn_embed = tf.keras.layers.Flatten()(tf.keras.layers.Embedding(8, 2, input_length=1)(regn))
    cais_grp_embed = tf.keras.layers.Flatten()(tf.keras.layers.Embedding(5, 2, input_length=1)(cais_grp))
    surf_grp_embed = tf.keras.layers.Flatten()(tf.keras.layers.Embedding(5, 2, input_length=1)(surf_grp))

    encode_slice = tf.keras.layers.Lambda(lambda x: x[:, :TIMESTEPS, :])

    x_in = tf.keras.layers.concatenate([seq_in, item_mean_in, store_mean_in, week_seq_in, is0_in, encode_slice(price_in), encode_slice(weekday_embed_encode)], axis=2)

    # Define network
    c1 = tf.keras.layers.Conv1D(latent_dim, 2, dilation_rate=1, padding='causal', activation='relu')(x_in)
    c2 = tf.keras.layers.Conv1D(latent_dim, 2, dilation_rate=2, padding='causal', activation='relu')(c1)
    c2 = tf.keras.layers.Conv1D(latent_dim, 2, dilation_rate=4, padding='causal', activation='relu')(c2)
    c2 = tf.keras.layers.Conv1D(latent_dim, 2, dilation_rate=8, padding='causal', activation='relu')(c2)
    c3 = tf.keras.layers.Conv1D(latent_dim, 2, dilation_rate=16, padding='causal', activation='relu')(c2)
    c3 = tf.keras.layers.Conv1D(latent_dim, 2, dilation_rate=32, padding='causal', activation='relu')(c3)
    c3 = tf.keras.layers.Conv1D(latent_dim, 2, dilation_rate=64, padding='causal', activation='relu')(c3)

    c4 = tf.keras.layers.concatenate([c1, c2, c3])

    conv_out = tf.keras.layers.Conv1D(8, 1, activation='relu')(c4)
    conv_out = tf.keras.layers.Dropout(0.25)(conv_out)
    conv_out = tf.keras.layers.Flatten()(conv_out)

    dnn_out = tf.keras.layers.Dense(512, activation='relu')(tf.keras.layers.Flatten()(x_in))
    dnn_out = tf.keras.layers.Dense(256, activation='relu')(dnn_out)
    dnn_out = tf.keras.layers.Dropout(0.25)(dnn_out)

    decode_slice = tf.keras.layers.Lambda(lambda x: x[:, TIMESTEPS:, :])
    price_pred = tf.keras.layers.Flatten()(decode_slice(price_in))

    x = tf.keras.layers.concatenate([conv_out, dnn_out, voct_embed, regn_embed, cais_grp_embed, surf_grp_embed])

    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    output_raw = tf.keras.layers.Dense(90, activation='relu')(x) #Relu activation because output >=0
    output = tf.keras.layers.Multiply()([output_raw, price_pred]) #Multiply prediction by on_sale sequence

    model = tf.keras.models.Model(inputs=[seq_in, week_seq_in, is0_in, price_in, item_mean_in, store_mean_in, weekday_in, cat_features], outputs=output)
    model.compile(optimizer='adam', loss=custom_loss)

    return model


def scheduler(epoch, lr):
    if epoch < 9:
        return lr
    elif epoch < 12:
        return lr/10
    else:
        return lr/100


test_pred = np.zeros((len(Xtest[0]), 90))
for bag in range(NBAGS):
    print("Training Model")
    model = get_model()
    callback = [tf.keras.callbacks.LearningRateScheduler(scheduler)]
    model.fit(train_set, steps_per_epoch=200, epochs=15, verbose=1, callbacks=callback)
    test_pred += model.predict(Xtest)


test_pred = test_pred / NBAGS
test_pred[:,0] = 0 #Set 1/1 sales to 0 because stores are closed. I tried a couple of different approaches (1 year lag, true predictions) here and this proved most effective

test = train.reset_index()[['ID_PDV','ID_ARTC']]
test_pred = pd.DataFrame(test_pred, columns=pd.date_range(date(2019,1,1), periods=90, freq='D'))
test = pd.concat([test, test_pred], axis=1)
sub = pd.melt(test, id_vars=['ID_PDV','ID_ARTC'], value_vars=pd.date_range(date(2019,1,1), periods=90, freq='D'),
              value_name='QTE', var_name='DATE')
sub['datestr'] = sub['DATE'].astype(str)
sub['id'] = sub['ID_PDV'].astype(str) + '_' + sub['ID_ARTC'].astype(str) + '_' + sub['datestr'].str.replace('-','')
sub_raw = sub.copy()

#Make sure to subtract our score indicator from the final predictions
sub['QTE'] = np.clip(np.round(np.expm1(sub['QTE']-score_ind)), 0, 100000).astype(int)
sub_raw['QTE'] = np.clip(np.expm1(sub_raw['QTE']-score_ind), 0, 100000)

sub = sub[['id','QTE']]
sub['QTE'] = sub['QTE'].replace(0, np.nan)
sub = sub[sub.QTE.notnull()].reset_index(drop=True)
sub.to_csv("output/submission_cnn.csv", index=False) #Submit for ~0.557
# 0.5573

sub_raw.to_csv("raw_predictions/cnn.csv", index=False, header=True) #Save non-rounded predictions for blending
