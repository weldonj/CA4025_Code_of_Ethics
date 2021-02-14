#!/usr/bin/env python
# coding: utf-8

# # Collaborative Filtering Recommender System

# ## Import Modules, create Spark session and load in the Dataset

# In[285]:


import pandas as pd
import pyspark.sql.functions as func
from pyspark.sql.types import DateType, IntegerType,NumericType
from pyspark.sql.functions import min, max, col
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.ml.feature import QuantileDiscretizer
from pyspark.ml.feature import Bucketizer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder,CrossValidatorModel

# Create our Spark Session
spark = SparkSession.builder.appName('recnn').getOrCreate()


# Once the spark session was created, it was time to read in the dataset from the HDFS as below

# In[286]:


df = spark.read.csv("hdfs://localhost:9000/john/hadoop/input/steam-200k.csv")


# ## Quick look at the Data and tidying
# 
# Now I wanted to take a quick look at the dataset and what the rows looked like

# In[287]:


df.head(15)


# And now a look at some basic stats, including the count of how many entries we have in the dataset (200,000) the maximum play time of any game by any user (999 hours) and the minimum play time of any game by any user (0.1 hours)

# In[288]:


df.describe().show()


# I now wanted to drop column 'c4' as it is not clear what it actually represents, and I also don't need whatever it may be for collaborative filtering

# In[290]:


df = df[('_c0', '_c1', '_c2', '_c3')]


# It was necessary to remove the rows that are for purchases instead of play times. These rows also weren't needed for my collaborative filtering, I just needed to know how much time each user spent playing each game.

# In[292]:


df = df[df['_c2'] != 'purchase']


# So I was now left with just our user ID, game title and how many hours they have played the game for

# In[293]:


df.head()


# I now created a 'rating' column that was initially just a duplicate of the play duration column, and also a 'rating_double' column which contained a double version of this instead of string

# In[297]:


df = df.withColumn('rating', df._c3)
changedTypedf = df.withColumn("rating_double", df["rating"].cast(DoubleType()))


# Looking at the Schema at this point, I now had a column containing the play times as doubles

# In[298]:


changedTypedf.printSchema()


# ## Create a derived Ratings column
# 
# The next six cells created a dataframe that gave me the total hours played by any single gamer (gamer_total_hours_played), and what percentage of that total is made up by each game they played (perc_of_gamer_total_hours).

# In[300]:


hours_by_game = changedTypedf.groupBy("_c0", "_c1").sum("rating_double")


# In[301]:


hours_by_gamer = changedTypedf.groupBy("_c0").sum("rating_double")


# In[302]:


hours_by_gamer = hours_by_gamer.withColumnRenamed("sum(rating_double)", "gamer_total_hours_played")


# In[303]:


hours_by_gamer = hours_by_gamer.withColumnRenamed("_c0", "gamer")


# In[304]:


hours_by_gamer_by_game = hours_by_game.alias("a").join(hours_by_gamer                     .alias("b"),hours_by_game['_c0'] == hours_by_gamer['gamer'],how='left')


# In[305]:


hours_by_gamer_by_game = hours_by_gamer_by_game.withColumn("perc_of_gamer_total_hours", 
                                                         hours_by_gamer_by_game["sum(rating_double)"]/ hours_by_gamer_by_game["gamer_total_hours_played"] )


# In[306]:


hours_by_gamer_by_game.show(10)


# Because I now had this information, I was then able to add a 'rank' column which will order, from 1 to n, the games played by each gamer in terms of how much time they spent playing them. 

# In[307]:


ranked =  hours_by_gamer_by_game.withColumn("rank", dense_rank().over(Window.partitionBy("gamer").orderBy(desc("perc_of_gamer_total_hours"))))


# In[308]:


ranked = ranked.withColumn("gamer_total_hours_played", func.round(ranked["gamer_total_hours_played"], 2))
ranked = ranked.withColumn("perc_of_gamer_total_hours", func.round(ranked["perc_of_gamer_total_hours"], 2))


# In[309]:


ranked.show(10)


# Above we can now the top 10 played games for gamer '108750879'

# In[310]:


new_ranked = ranked.drop("sum(rating_double)")


# In[311]:


new_ranked.show(20)


# The Spark ML function, 'QuantileDiscretizer' allowed me to assign a rating (0-9) from each gamer for each game, by separating the games they played into buckets based on the time spent playing. 
# 
# The game that they spent the most time playing would receive a 9 for example.

# In[312]:


game_rated = QuantileDiscretizer(numBuckets=10, inputCol="perc_of_gamer_total_hours",outputCol="rating").fit(hours_by_gamer_by_game).transform(hours_by_gamer_by_game)


# In[313]:


ratings = game_rated.select("gamer","_c1", "rating","perc_of_gamer_total_hours")


# In[314]:


ratings = ratings.withColumnRenamed("_c1", "game")


# In[315]:


ratings.show(20)


# In[316]:


ratings = ratings.withColumn("gamer", ratings["gamer"].cast(IntegerType()))


# In[317]:


ratings.show(10)


# It can be seen above that I now had the rating from each gamer for each game. I needed a game_id column for the model as it doesn't take in strings and so this next cell achieved this. It gave a unique game_id from 1 to n (n being the total number of games) to each game in the list

# In[318]:


ratings = ratings.withColumn("game_id", func.dense_rank().over(Window.orderBy(ratings.game)))


# ## Build and train the Model
# 
# I was now able to split these ratings into training and test sets

# In[319]:


training, test = ratings.randomSplit([0.8,0.2], seed = 36)


# Next I defined the alternating least squares model for collaborative filtering

# In[320]:


from pyspark.ml.recommendation import ALS


als = ALS(maxIter=10, regParam=0.01, userCol="gamer", itemCol="game_id", ratingCol="rating",
          coldStartStrategy="drop", nonnegative = True)


# Fit the model to the training set

# In[321]:


model = als.fit(training)


# Use the model to make predictions on the test set

# In[322]:


predictions = model.transform(test)


# In[323]:


predictions.show(20)


# ## Evaluate the Model

# In[325]:


evaluator = RegressionEvaluator(metricName='rmse', labelCol='rating')
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))


# Having evaluated the performance using root mean square error, I could see it was around 3. This isn't amazing for a range of 0 to 9 but it would still likely produce useful recommendations

# ## Recommendations

# It was now time to get some actual recommendations based on the model. First I wanted to get the top 20 gamers in terms of hours played which I did in the next cell

# In[333]:


top_gamers = changedTypedf.groupBy(changedTypedf['_c0']).agg({'_c3':"sum"}).sort("sum(_c3)", ascending=False).dropna().limit(20)
top_gamers.show(20)


# I placed these gamers into a list that I could then use

# In[335]:


top20_gamer_list = [row._c0 for row in top_gamers.select("_c0").collect()]
top20_gamer_list


# Using this list and the model, I created a dictionary containing all 20 top gamers and the top 10 game recommendations for each

# In[337]:


gamer_recs = model.recommendForAllUsers(10)

rec_game_top20_gamers = {}

for gamer in top20_gamer_list:
    rec_game = gamer_recs.where(gamer_recs.gamer == gamer).select("recommendations").collect()
    rec_game_top20_gamers[gamer] = [i.game_id for i in rec_game[0]["recommendations"]]

rec_game_top20_gamers


# I wanted a list of game titles and their game ids so that I could see what the actual game recommendations were

# In[331]:


game_list = ratings.select('game_id', 'game').dropDuplicates()
game_list.show(20)


# I took one of the gamers as user1, (gamer id '10599862') and had a look at the recommendations and also the actual games they played the most of

# In[339]:


user1_recommend = game_list.filter(game_list["game_id"].isin(rec_game_top20_gamers['10599862']))
user1_recommend


# In[340]:


user1_recommend.show()


# In[352]:


ratings.filter(ratings.gamer=="10599862").sort('rating', ascending=False).limit(10).show()


# I could see that the recommendations passed a basic sanity test at least, Sports games and Fantasy games were being recommended to user1 and I could see that similar games appear in their top played games

# In[342]:


user2_recommend = game_list.filter(game_list["game_id"].isin(rec_game_top20_gamers['100630947']))
user2_recommend.show()


# In[353]:


ratings.filter(ratings.gamer=="100630947").sort('rating', ascending=False).limit(10).show()


# In[349]:


user3_recommend = game_list.filter(game_list["game_id"].isin(rec_game_top20_gamers['26762388']))
user3_recommend.show()


# In[351]:


ratings.filter(ratings.gamer=="26762388").sort('rating', ascending=False).limit(10).show()


# I did the same for two more users and again I was happy enough with the reccomendations. At this point I was fairly confident that the system was doing its job and likely making some useful recommendations
