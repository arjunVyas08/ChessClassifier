# Training a Chess move predictor using a Classifier model

## Introduction

### Scope 
The end goal of this project is to produce a move prediction classifier, that when given a random position ranks each move by how likely it is that a strong human plays that move.

data set: lichess_db_standard_rated_2026-09.pgn.zst

This project will not include:
    - A gui
    - Self play
    - Reinforcement learning

### Requirements to replicate
        pandas , numpy, python-chess , scikit-learn , zstandard

        September 2026 lichess data base :lichess_db_standard_rated_2026-09.pgn.zst, Found on the lichess database webpage (https://database.lichess.org/)

        Python 3 +, any version of python 3 that is compatable with the libraries above


        The classifier will recieve a position and classify it into a space that corresponds to a move

### Motivations and Past experience
This is a side project with no goal beyond learning.

For this project I will be using the September 2026 Lichess dataset, this was done as in a previous attempt I had used the Kaggle chess csv table. This table contained approximately 20,000 games and when I wanted to apply masks to filter out low rated games (below 1500 elo) or games that were not rated the number of games dropped drastically to approximately 3000 games. This was not ideal as to ensure there was sufficient training data I would have to lower the quality of the training data which was not desirable.

The Database that I have chosen is the 2026 September database, this contains every game that was played in the month of september which is a vast increase over the kaggle dataset. An advantage of this is that I can Apply more restrictive masks and end up with a lot more High quality data which will allow for more training and testing data. For example, I can set the minimum elo for each player to 2000+ elo and also set a limit for the elo gap between the two players, to ensure that the games aren't lopsided, without having to compromise these to actually have a useable amount of data. In the previous attempt, I have gone from 16k games to 3k games by applying the following mask:

``` py 
mask = (df["white_rating"] >= 1600) & (df["black_rating"] >= 1600) & (abs(df["white_rating"] - df["black_rating"]) <= 100)
```
I was forced to compromise these restrictions for the reason that too much of the data was removed, and this was before any restriction was placed on the time control. At the time I intended to drop sub 5-minute games as well, had I done so the surviving set would have been even smaller, which is part of why I abandoned that filter.


I have learnt from my previous attempt that I should plan out every step before hand, to avoid ending up mid preprocessing without a plan of what to do next, it is also a good opportunity to practice my write up skills


For this current step (preprocessing) I need to filter out games that wouldnt be useful to training the model, namely games with poor play , either from low elo players or players low on time. Since the data set covers millions of games we can use a restriction like both players have to be above 2200 elo and not have an elo gap of above 100. We can also filter out unrated games as these are less likely to be played seriously.