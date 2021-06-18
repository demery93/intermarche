'''
Global lightgbm model with tweedie loss function.
This was an approach that surprisingly worked for me in the M5 forecasting competition.
It only uses information > 90 days old, so we can predict all days in the forecast horizon at once.
'''

import pandas as pd
import numpy as np
import gc
from datetime import date
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

score_ind = np.log(2) / 2 #Score indicator
train = pd.read_csv("input/ventes_2018.csv",
                    parse_dates=['DATE'],
                    converters={'QTE': lambda u: np.log1p(float(u)) if float(u) > 0 else 0})

print("Filling all nans with 0")
print(train.shape)
train = pd.pivot_table(train, index=['ID_PDV','ID_ARTC'], columns=['DATE'], values=['QTE']).fillna(0)
train.columns = train.columns.get_level_values(1)

scored = (train > 0).astype(int)
for i in range(1, 7):
    scored = scored + scored.shift(i, axis=1).fillna(0)
scored = (scored >= 1).astype(int) * score_ind

train = train + scored #Increment training data by indicator

#Create test dates
for c in pd.date_range(date(2019,1,1), periods=90):
    train[c] = np.nan

train.reset_index(inplace=True)
train = pd.melt(train, id_vars=['ID_PDV','ID_ARTC'], value_vars=train.columns[2:], value_name='QTE', var_name='DATE')
print(train.shape)

## Generate Calendar Features
train['year'] = train['DATE'].dt.year # Year won't help but I forgot to remove it
train['quarter'] = train['DATE'].dt.quarter
train['month'] = train['DATE'].dt.month # Month is used to group training / validation
train['day_of_month'] = train['DATE'].dt.day
#train['week'] = train['DATE'].dt.week
train['day_of_week'] = train['DATE'].dt.weekday
train['holiday'] = train.DATE.isin([date(2018,1,1), date(2018,12,25), date(2019,1,1)]).astype(int)

products = pd.read_csv("input/nomenclature_produits.csv")
region = pd.read_csv("input/points_de_vente.csv")
price = pd.read_csv("input/prix_vente.csv")
price['on_sale'] = 1

train = train.merge(price, left_on=['ID_PDV','ID_ARTC','year','quarter'], right_on=['ID_PDV','ID_ARTC','ANNEE','TRIMESTRE'], how='left')
train.drop(['ANNEE','TRIMESTRE','PRIX_UNITAIRE','quarter'], axis=1, inplace=True)
train['on_sale'] = train['on_sale'].fillna(0)

#Merge Product Information
train = train.merge(products, on='ID_ARTC', how='left')
train = train.merge(region, on='ID_PDV', how='left')
for c in ['LB_VENT_RAYN','LB_VENT_FAML','LB_VENT_SOUS_FAML','ID_VOCT','ID_REGN','NB_CAIS_GRP','SURF_GRP']:
    train[c] = LabelEncoder().fit_transform(train[c])

del price, products, region
gc.collect()

train['lag'] = train.groupby(['ID_ARTC','ID_PDV'])['QTE'].transform(lambda x: x.shift(91))
train['on_sale_lag'] = train.groupby(['ID_ARTC','ID_PDV'])['on_sale'].transform(lambda x: x.shift(91))

#Generate rolling means and standard deviations for 4 time windows
for window in [7, 14, 28, 91]:
    print(f"Generating Features for window {window}")
    train['mean_' + str(window)] = train.groupby(['ID_ARTC','ID_PDV'])['lag'].transform(lambda x: x.rolling(window).mean())
    train['std_' + str(window)] = train.groupby(['ID_ARTC','ID_PDV'])['lag'].transform(lambda x: x.rolling(window).std())

print(f"Training shape before filtering: {train.shape}")

'''
In the other two scripts, the loss function was key to incorporating the scoring mechanic.
For this script, we can simply filter out the unscored records, so we don't need a custom loss function
'''
## Filter 1 - Remove first 182 days of data for na values
train = train[train.mean_91.notnull()].reset_index(drop=True)
print(f"Training shape after filter 1: {train.shape}")

## Filter 2 - We know when on_sale == 0, then qte == 0, so no need to train on this
train = train[train.on_sale == 1].reset_index(drop=True)
print(f"Training shape after filter 2: {train.shape}")

## Filter 3 - Remove all data where qte == 0, this data is not scores, so we do not need to train on it
train = train[train.QTE != 0].reset_index(drop=True)
train['QTE'] = train['QTE'] - score_ind #Reset scored zeros to 0
print(f"Training shape after filter 3: {train.shape}")

# Pretty random hyperparameters. I tried a couple different tweedie_variance_power values but 1.1 seems okay
params = {
    'boosting_type': 'gbdt',
    'objective': 'tweedie',
    'tweedie_variance_power': 1.1,
    'metric': 'rmse',
    'subsample': 0.5,
    'subsample_freq': 1,
    'learning_rate': 0.1,
    'num_leaves': 2 ** 11 - 1,
    'min_data_in_leaf': 2 ** 12 - 1,
    'feature_fraction': 0.5,
    'max_bin': 100,
    'boost_from_average': False,
    'verbose': -1
}

n_splits = 5
kfold = GroupKFold(n_splits=n_splits)
test = train[train.QTE.isnull()].reset_index(drop=True)
train = train[train.QTE.notnull()].reset_index(drop=True)
group = train['month']
y = train['QTE'].values
features = [c for c in train.columns if c not in ['QTE','DATE','year','month','on_sale']]
oof = np.zeros(len(train))
oof2 = np.zeros(len(train))
test_pred = np.zeros(len(test))
for i, (train_idx, val_idx) in enumerate(kfold.split(train, y, groups=group)):
    x_train, y_train = train[features].iloc[train_idx], y[train_idx]
    x_val, y_val = train[features].iloc[val_idx], y[val_idx]

    print(f"Beginning fold {i+1}: {x_train.shape, x_val.shape}")

    dtrain = lgb.Dataset(x_train, y_train)
    dval = lgb.Dataset(x_val, y_val)
    model = lgb.train(params, dtrain, num_boost_round=500, early_stopping_rounds=20,
                  valid_sets=[dtrain, dval], verbose_eval=10)

    # predict out of fold and test with true price data
    oof[val_idx] = model.predict(x_val, num_iteration=model.best_iteration)
    test_pred += model.predict(test[features], num_iteration=model.best_iteration) / n_splits

    print(np.sqrt(np.mean((oof[val_idx] - y[val_idx])**2)))

oof_real = np.round(np.clip(np.expm1(oof), 0, 100000))
y_real = np.round(np.expm1(y))
score = np.round(np.sqrt(np.mean((np.log1p(y_real) - np.log1p(oof_real))**2)),3)
print(f"CV RMSLE: {score}")

sub = test[['ID_PDV','ID_ARTC','DATE']].copy()
sub['datestr'] = sub['DATE'].astype(str)
sub['id'] = sub['ID_PDV'].astype(str) + '_' + sub['ID_ARTC'].astype(str) + '_' + sub['datestr'].str.replace('-','')
sub['QTE'] = np.clip(np.round(np.expm1(test_pred)), 0, 100000).astype(int)

sub = sub[['id','QTE']]
sub['QTE'] = sub['QTE'].replace(0, np.nan)
sub = sub[sub.QTE.notnull()].reset_index(drop=True)
sub.to_csv(f"output/submission_kfold_tweedie_lightgbm_{score}.csv", index=False) #Submission. Scores 0.5673

sub = test[['ID_PDV','ID_ARTC','DATE']].copy()
sub_raw = sub.copy()
sub_raw['QTE'] = np.clip(np.expm1(test_pred), 0, 100000)
sub_raw.to_csv("raw_predictions/lgb_tweedie.csv", index=False, header=True) #Save raw predictions
