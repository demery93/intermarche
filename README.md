# intermarche
5th place solution to intermarche competition.

TLDR: My solution is a blend of a convolutional neural net, simple feed forward neural net, and a global lightgbm model.

<h3>Data</h3>
Given 1 year of sales data, we were asked to predict the following 90 days of each item-store combination. I filled in all 0-sales observations and then identified which 0s were scored. 

Each model uses different features as inputs and should be seen as a completely different approach.

<h3>Models</h3>
<b>CNN</b>:
This was my best performing model throughout the competition. I input the log transformed sales sequence, item mean sequence, store mean sequence, price sequence, and categorical features through embedding layers to produce all 90 predictions at once. My model architecture is inspired by Lenz Du's solution to a previous Kaggle competition (https://github.com/LenzDu/Kaggle-Competition-Favorita) and is based off of Wavenet. The trick to get this model to work for this competition was to increment my scored training values by ln(2)/2 and then use a custom loss to ignore observations equal to 0. This mimics the competition's metric.


<b>MLP</b>:
Here I simply generate a series of rolling window statistics for the item-store, item, and store levels and passed them through an Multilayered Perceptron with the same custom loss and scoring concept from the CNN. This model was pretty volitile and individually scored from 0.5599 - 0.5640.


<b>LightGBM</b>:
This model is built of 91 day lagged statistics (means and standard deviations), categorical features, and calendar features. I keep scored zeros for training and remove all non-scored zeros. I then use grouped 5 fold cross validation, grouped by month, to train 5 models and predict the entire 90 days at once. Because I removed non-scored zeros, there is no need for a custom loss here, however, I achieved slightly better results with a tweedie loss function here.

<b>Final Blend</b>:
0.5*cnn + 0.25*mlp + 0.25*lgb -> score 0.556

1.04*(0.5*cnn + 0.25*mlp + 0.25*lgb) -> score 0.554

Multiplying my final predictions did improve the score quite a bit, but this likely means my models were not picking up on a trend. I would not have done something like this if there was a private leaderboard.

<h3>Conclusion</h3>
I'd like to thank everyone at intermarche who made this competition possible. It was really fun to work on and learn new approaches. I hope to see more competitions from this team in the future and hopefully I will be prize eligible in them :)
