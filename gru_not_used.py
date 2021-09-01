'''
This script was not included in my final solution
It is a sequence to sequence gru that predicts 7 days at a time
for 12 weeks and then the final 6 days to complete the prediction
'''

import pandas as pd
import numpy as np

import tensorflow as tf
import gc
import datetime

from datetime import timedelta, date
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, StandardScaler

score_ind = np.log(2)/2
NBAGS = 3
TIMESTEPS = 100
train = pd.read_csv("input/ventes_2018.csv",
                    parse_dates=['DATE'],
                    converters={'QTE': lambda u: np.log1p(float(u)) if float(u) > 0 else 0})

region = pd.read_csv("input/points_de_vente.csv")
products = pd.read_csv("input/nomenclature_produits.csv")
price = pd.read_csv("input/prix_vente.csv")
price['low_end'] = price['PRIX_UNITAIRE'].str[6:8]
price['low_end'] = price['low_end'].replace('de', '0').astype(int)
price['high_end'] = price['PRIX_UNITAIRE'].str[-6:-1].astype(float)
price['price_numeric'] = (price.low_end + price.high_end ) / 2
train['quarter'] = train['DATE'].dt.quarter
test_price = price[price.ANNEE == 2019].reset_index(drop=True)
test_price = pd.pivot_table(test_price, index=['ID_PDV','ID_ARTC'], columns=['TRIMESTRE'], values=['price_numeric'])
test_price.columns = test_price.columns.get_level_values(1)
fillval = test_price.values
test_price = pd.DataFrame(index=test_price.index, columns = [c for c in pd.date_range(date(2019,1,1), periods=91, freq='D')])
for c in pd.date_range(date(2019,1,1), periods=91, freq='D'):
    test_price[c] = fillval

train = train.merge(price[price.ANNEE == 2018][['ID_PDV','ID_ARTC','TRIMESTRE','price_numeric']], left_on=['ID_PDV','ID_ARTC','quarter'], right_on=['ID_PDV','ID_ARTC','TRIMESTRE'], how='left')

price = pd.pivot_table(train, index=['ID_PDV','ID_ARTC'], columns=['DATE'], values=['price_numeric'])
price.columns = price.columns.get_level_values(1)
print(train.shape, price.shape, test_price.shape)

train = pd.pivot_table(train, index=['ID_PDV','ID_ARTC'], columns=['DATE'], values=['QTE']).fillna(0)
train.columns = train.columns.get_level_values(1)
cat_features = train.reset_index()[['ID_PDV']].merge(region, how='left', on='ID_PDV').drop(['ID_PDV'], axis=1)
cat_features2 = train.reset_index()[['ID_ARTC']].merge(products, how='left', on='ID_ARTC').drop(['ID_ARTC'], axis=1)
cat_features = pd.concat([cat_features, cat_features2], axis=1)
for c in cat_features.columns:
    print(c, len(cat_features[c].unique()))

price = train.reset_index()[['ID_PDV','ID_ARTC']].merge(price.reset_index(), how='left', on=['ID_PDV','ID_ARTC'])
test_price = train.reset_index()[['ID_PDV','ID_ARTC']].merge(test_price.reset_index(), how='left', on=['ID_PDV','ID_ARTC'])

price = pd.concat([price, test_price.drop(['ID_PDV','ID_ARTC'], axis=1)], axis=1)
price.set_index(['ID_PDV','ID_ARTC'], inplace=True)
price = price.ffill(axis=1)
price = price.bfill(axis=1)
price = np.log1p(price)
price.columns = [pd.to_datetime(c) for c in price.columns]

## Identifying scored records and incrementing by 1 in prep of custom rmse loss
scored = (train > 0).astype(int)
for i in range(1,7):
    scored = scored + scored.shift(i, axis=1).fillna(0)
scored = (scored >= 1).astype(int) * score_ind
train = train + scored

def get_timespan(df, pred_start, timesteps=TIMESTEPS, is_train=True):
    X = df[pd.date_range(pred_start-timedelta(days=timesteps), pred_start-timedelta(days=1))].values
    if is_train: y = df[pd.date_range(pred_start, periods=90)].values
    else: y = None
    return X, y

def train_generator(df, price, cat_features, first_pred_start, batch_size=2000, n_range=20, day_skip=1):
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
            price_tmp = price.iloc[keep_idx, :]
            item_tmp = item_group_mean.iloc[keep_idx,:]
            store_tmp = store_group_mean.iloc[keep_idx,:]
            cat_tmp = cat_features[keep_idx]

            pred_start = first_pred_start + timedelta(days=int(day_skip*i))
            yield create_dataset_part(df_tmp, price_tmp, item_tmp, store_tmp, cat_tmp, pred_start, True)
            gc.collect()

def create_dataset(df, price, cat_features, pred_start, is_train=True):
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
    return create_dataset_part(df, price, item_group_mean, store_group_mean, cat_features, pred_start, is_train)

def create_dataset_part(df, price, item_group_mean, store_group_mean, cat_features, pred_start, is_train=True):
    x, y = get_timespan(df, pred_start,timesteps=TIMESTEPS, is_train=is_train)
    x_price, _ = get_timespan(price, pred_start+timedelta(91), timesteps=TIMESTEPS+90+1, is_train=False)

    x_item, _ = get_timespan(item_group_mean, pred_start, is_train=False)
    x_store, _ = get_timespan(store_group_mean, pred_start, is_train=False)

    weekday = np.tile([d.weekday() for d in pd.date_range(pred_start - timedelta(days=TIMESTEPS), periods=TIMESTEPS + 90 + 1)],(x.shape[0], 1))

    x = x.reshape((-1, TIMESTEPS, 1))
    x_price = x_price.reshape((-1, TIMESTEPS+90+1, 1))
    x_item = x_item.reshape((-1, TIMESTEPS, 1))
    x_store = x_store.reshape((-1, TIMESTEPS, 1))

    cat_features = np.tile(cat_features[:, None, :], (1, TIMESTEPS + 90 + 1, 1))

    return([x, x_price, x_item, x_store, weekday, cat_features], y)

n_range = 24
train_pred_start = date(2019,1,1) - timedelta(91) - timedelta(7*n_range)
print(f"Training sequences start {train_pred_start - timedelta(TIMESTEPS)}")
print(f"Training predictions start {train_pred_start}")
print(f"Training predictions end {train_pred_start + timedelta(7*n_range + 90)}")


train_set = train_generator(train, price, cat_features, first_pred_start=train_pred_start, day_skip=7, n_range=n_range)
# Fill Christmas and Christmas Eve with Naive data to combat irregular demand pattern
train['2018-12-25'] = train['2018-12-18'].values #Smoothing Christmas
train['2018-12-24'] = train['2018-12-17'].values #Smoothing Christmas Eve
Xtest, _ = create_dataset(train, price, cat_features, date(2019,1,1), is_train=False)

def custom_loss(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true[y_true != 0] - y_pred[y_true != 0])))

def get_model():
    latent_dim=50
    seq_in = tf.keras.layers.Input(shape=(TIMESTEPS, 1))
    price_in = tf.keras.layers.Input(shape=(TIMESTEPS+90+1, 1))
    weekday_in = tf.keras.layers.Input(shape=(TIMESTEPS + 90 + 1,), dtype='uint8')
    item_mean_in = tf.keras.layers.Input(shape=(TIMESTEPS, 1))
    store_mean_in = tf.keras.layers.Input(shape=(TIMESTEPS, 1))

    cat_features = tf.keras.layers.Input(shape=(TIMESTEPS+90+1, 6))

    voct = tf.keras.layers.Lambda(lambda x: x[:, :, 0])(cat_features)
    regn = tf.keras.layers.Lambda(lambda x: x[:, :, 1])(cat_features)
    cais_grp = tf.keras.layers.Lambda(lambda x: x[:, :, 2])(cat_features)
    surf_grp = tf.keras.layers.Lambda(lambda x: x[:, :, 3])(cat_features)


    weekday_embed_encode = tf.keras.layers.Embedding(7, 4, input_length=TIMESTEPS + 90 + 1)(weekday_in)
    voct_embed = tf.keras.layers.Embedding(5, 2, input_length=TIMESTEPS+90+1)(voct)
    regn_embed = tf.keras.layers.Embedding(8, 2, input_length=TIMESTEPS+90+1)(regn)
    cais_grp_embed = tf.keras.layers.Embedding(5, 2, input_length=TIMESTEPS+90+1)(cais_grp)
    surf_grp_embed = tf.keras.layers.Embedding(5, 2, input_length=TIMESTEPS+90+1)(surf_grp)

    encode_slice = tf.keras.layers.Lambda(lambda x: x[:, :TIMESTEPS, :])
    encode_features = tf.keras.layers.concatenate([price_in, weekday_embed_encode,
                                   voct_embed, regn_embed, cais_grp_embed, surf_grp_embed], axis=2)

    conv_in = tf.keras.layers.Conv1D(4, 5, padding='same')(seq_in)

    x_encode = tf.keras.layers.concatenate([seq_in, conv_in, item_mean_in, store_mean_in, encode_slice(encode_features)], axis=2)

    encoder = tf.keras.layers.GRU(latent_dim, return_state=True)
    print('Input dimension:', x_encode.shape)
    _, h = encoder(x_encode)

    # Connector
    h = tf.keras.layers.Dense(latent_dim, activation='tanh')(h)

    # Decoder
    decode_slice = tf.keras.layers.Lambda(lambda x: x[:, TIMESTEPS:, :])
    decode_features = tf.keras.layers.concatenate([price_in, weekday_embed_encode,
                                                   voct_embed, regn_embed, cais_grp_embed, surf_grp_embed], axis=2)
    decode_features = decode_slice(decode_features)
    previous_x = tf.keras.layers.Lambda(lambda x: x[:, -7:, :])(seq_in)

    decoder = tf.keras.layers.GRU(latent_dim, return_state=True, return_sequences=False)

    decoder_dense2 = tf.keras.layers.Dense(7, activation='relu')
    decoder_dense3 = tf.keras.layers.Dense(6, activation='relu')
    slice_at_t = tf.keras.layers.Lambda(lambda x: tf.slice(x, [0, 7*i, 0], [-1, 7, -1]))
    for i in range(13):
        previous_x = tf.keras.layers.Reshape((7, 1))(previous_x)
        features_t = slice_at_t(decode_features)

        decode_input = tf.keras.layers.concatenate([previous_x, features_t], axis=2)
        output_x, h = decoder(decode_input, initial_state=h)

        # gather outputs
        if i == 0:
            output_x = decoder_dense2(output_x)
            decoder_outputs = output_x
        elif i < 12:
            output_x = decoder_dense2(output_x)
            decoder_outputs = tf.keras.layers.concatenate([decoder_outputs, output_x])
        else:
            output_x = decoder_dense3(output_x)
            decoder_outputs = tf.keras.layers.concatenate([decoder_outputs, output_x])

        previous_x = output_x


    model = tf.keras.models.Model(inputs=[seq_in, price_in, item_mean_in, store_mean_in, weekday_in, cat_features], outputs=[decoder_outputs])
    model.compile(optimizer='adam', loss=custom_loss)
    return model

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

print("Training and predicting models...")
model = get_model()
def scheduler(epoch, lr):
    if epoch < 13:
        return lr
    elif epoch < 17:
        return lr/10
    else:
        return lr/100
callback = [tf.keras.callbacks.LearningRateScheduler(scheduler)]
models = []
for i in range(NBAGS):
    model = get_model()
    model.fit(train_set, steps_per_epoch=200, epochs=20, verbose=1, callbacks=callback)
    models.append(model)


#0.4772

test_pred = np.zeros((len(Xtest[0]), 90))
for model in models:
    test_pred += model.predict(Xtest) / NBAGS
test_pred[:,0] = train.values[:,0]

test = train.reset_index()[['ID_PDV','ID_ARTC']]
test_pred = pd.DataFrame(test_pred, columns=pd.date_range(date(2019,1,1), periods=90, freq='D'))
test = pd.concat([test, test_pred], axis=1)
sub = pd.melt(test, id_vars=['ID_PDV','ID_ARTC'], value_vars=pd.date_range(date(2019,1,1), periods=90, freq='D'),
              value_name='QTE', var_name='DATE')
sub['datestr'] = sub['DATE'].astype(str)
sub['id'] = sub['ID_PDV'].astype(str) + '_' + sub['ID_ARTC'].astype(str) + '_' + sub['datestr'].str.replace('-','')
sub['QTE'] = np.clip(np.round(np.expm1(sub['QTE']-score_ind)), 0, 100000).astype(int)

sub = sub[['id','QTE']]
sub['QTE'] = sub['QTE'].replace(0, np.nan)
sub = sub[sub.QTE.notnull()].reset_index(drop=True)
sub.to_csv(f"output/submission_full_gru_custom_loss_bagged.csv", index=False)
