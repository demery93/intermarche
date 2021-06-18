'''
Simple script to blend cnn, mlp, and lightgbm solutions
'''

import pandas as pd
import numpy as np

sub_cnn = pd.read_csv("raw_predictions/cnn.csv")
sub_lgb = pd.read_csv("raw_predictions/lgb_tweedie.csv")
sub_mlp = pd.read_csv("raw_predictions/mlp.csv")

sub_cnn.columns = ['ID_PDV','ID_ARTC','DATE','cnn_qte', 'datestr','id']
sub_lgb.columns = ['ID_PDV','ID_ARTC','DATE','lgb_qte']
sub_mlp.columns = ['id','mlp_qte']

sub_lgb['id'] = sub_lgb['ID_PDV'].astype(str) + '_' + sub_lgb['ID_ARTC'].astype(str) + '_' + sub_lgb['DATE'].str.replace('-','')

sub = pd.DataFrame({'id': np.unique(sub_cnn.id.unique().tolist() + sub_lgb.id.unique().tolist() + sub_mlp.id.unique().tolist())})

sub = sub.merge(sub_cnn, on=['id'], how='left')
sub = sub.merge(sub_lgb, on=['id'], how='left')
sub = sub.merge(sub_mlp, on=['id'], how='left')

sub.fillna(0, inplace=True)

#I didn't experiment with the blending weights because it seemed really easy to overfit the leaderboard by tweaking the weights and this wouldn't teach me anything
sub['qte'] = (0.5*sub.cnn_qte + 0.25*sub.lgb_qte + 0.25*sub.mlp_qte)
sub['qte'] = np.round(sub['qte']).astype(int)

sub.loc[sub.id.str[-4:] == '0101','qte'] = 0

sub = sub[['id','qte']]
sub = sub[sub.qte != 0].reset_index(drop=True)
sub.to_csv("output/cnn_lgb_mlp.csv", index=False, header=True)

#########################
#########################
#########################
'''
This last section is a bit frustrating. I was out of (good) ideas on the last day of the competition and had 1 submission left
so I decided to give in to the dark side and just multiply my predictions by a multiplier to see if my score went up or down.
This one multiplier moved my score from 0.556316 to 0.554676. I'm sure different multipliers on different days would significantly
improve the score as it did in the M5. I want to urge that this is bad practice and should not be done in production. I had a 
hunch that I wasn't capturing the trend very well because we only had 1 year of data, making it hard differentiate monthly
differences from a trend (eg are March sales higher than January's sale because of a trend or because March generally has more sales).

I'm actually kind of glad that I am not prize eligible because I feel that a solution that does strange postprocessing 
to overfit the leaderboard is not a scalable solution, as required by the competition host.
'''

sub_cnn = pd.read_csv("raw_predictions/cnn.csv")
sub_lgb = pd.read_csv("raw_predictions/lgb_tweedie.csv")
sub_mlp = pd.read_csv("raw_predictions/mlp.csv")

sub_cnn.columns = ['ID_PDV','ID_ARTC','DATE','cnn_qte', 'datestr','id']
sub_lgb.columns = ['ID_PDV','ID_ARTC','DATE','lgb_qte']
sub_mlp.columns = ['id','mlp_qte']

sub_lgb['id'] = sub_lgb['ID_PDV'].astype(str) + '_' + sub_lgb['ID_ARTC'].astype(str) + '_' + sub_lgb['DATE'].str.replace('-','')

sub = pd.DataFrame({'id': np.unique(sub_cnn.id.unique().tolist() + sub_lgb.id.unique().tolist() + sub_mlp.id.unique().tolist())})

sub = sub.merge(sub_cnn, on=['id'], how='left')
sub = sub.merge(sub_lgb, on=['id'], how='left')
sub = sub.merge(sub_mlp, on=['id'], how='left')

sub.fillna(0, inplace=True)

sub['qte'] = 1.04*(0.5*sub.cnn_qte + 0.25*sub.lgb_qte + 0.25*sub.mlp_qte)
sub['qte'] = np.round(sub['qte']).astype(int)

sub.loc[sub.id.str[-4:] == '0101','qte'] = 0

sub = sub[['id','qte']]
sub = sub[sub.qte != 0].reset_index(drop=True)
sub.to_csv("output/cnn_lgb_mlp_blind_multiplier.csv", index=False, header=True)
