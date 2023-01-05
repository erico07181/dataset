# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from matplotlib import rcParams
from IPython.core.interactiveshell import InteractiveShell
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, CrossValidatorModel
from pyspark.mllib.evaluation import RegressionMetrics
from pyspark.ml.regression import LinearRegression
from pyspark.sql.functions import udf, col
import pyspark.sql.functions as F
from pyspark.sql.types import *
from pyspark.sql import SparkSession, SQLContext
from pyspark import SparkConf, SparkContext
import numpy as np
import pandas as pd
import os
from IPython import get_ipython

# %%
get_ipython().system('pip install pyspark')


# %%


# %%


# %%
# Visualization
InteractiveShell.ast_node_interactivity = "all"

pd.set_option('display.max_columns', 200)
pd.set_option('display.max_colwidth', 400)

sns.set(context='notebook', style='whitegrid', rc={'figure.figsize': (18, 4)})
rcParams['figure.figsize'] = 18, 4

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# %%
# setting random seed for notebook reproducability
rnd_seed = 123
np.random.seed = rnd_seed
np.random.set_state = rnd_seed


# %%
df = '../input/irisdata/iris.data'


# %%
spark = SparkSession.builder.master("local[2]").appName(
    "Linear-Regression-IRIS").getOrCreate()


# %%
spark


# %%
iris_df = spark.read.csv(path=df).cache()


# %%
# Inspect first ten rows
iris_df.take(10)


# %%
# Show first ten rows
iris_df.show(10)


# %%
# show the dataframe columns
iris_df.columns


# %%
# define the schema, corresponding to a line in the csv data file.
schema = StructType([
    StructField("sepalLength", FloatType(), nullable=True),
    StructField("sepalWidth", FloatType(), nullable=True)]
)


# %%
# Load data
iris_df = spark.read.csv(path=df, schema=schema).cache()


# %%
# Inspect first ten rows
iris_df.take(10)


# %%
# run a sample selection
iris_df.select('sepalWidth', 'sepalLength').show(10)


# %%
iris_df.toPandas().plot.bar(x='sepalLength', figsize=(14, 6))


# %%
iris_df.toPandas().plot.bar(x='sepalWidth', figsize=(14, 6))


# %%
(iris_df.describe().select(
    "summary",
    F.round("sepalLength", 4).alias("sepalLength"),
    F.round("sepalWidth", 4).alias("sepalWidth"))
 .show())
