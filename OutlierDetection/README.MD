# Outlier Detection: HDBScan and single cluster DTW-KMeans
Using a combination of HDBScan and DTW-KMeans to create a upper tier of outlier detection on data with much variance. HDBScan is firstly used to limit the dataset to a high upper percentile to facilitate the use of DTW-KMeans with a single cluster. DTW-KMeans with a single cluster and a distance metric hereby differentiates between datapoints of similar character close to the KMeans center and datapoints regarded as different than center-mean.

HDBScan is used to limit the memory usage usually found in the SKLearn DBScan algorithme as it computes the full O(n^2) distance matrix.

<br>
<br>

# GAN for rare event outlier detection
This GAN makes of for the relative small number of outliers/targets found in some fraudulent interactions. It facilitates the use of a very imbalanced dataset as it synthesize new outlier values which is possible to further train on.

An issue with this kind of GAN on tabular data is its tendency to get a mode collapse, essentially meaning the generator starts producing a similar set of output every time, which "fools" the discriminator.

Credits: This GAN is heavily inspired by the book Machine Learning for Finance: Principles and practice for financial insiders by Jannes Klaas
  
<br>
<br>

# GradientBoostingClassifier
Template for setting up a rudimentary GradientBoostingClassifier for imbalanced dataset using SMOTE able to detect outliers with a high percentage of accuracy.

### basic_parameter_tuning.py:
Used to finetune parameters in isolation for GradientBoostingClassifier. In high dimensional data with a large quantity of datapoints, use the finetuning isolated with other parameters being default and test the combinations in the end.

### gradientboostingclassifier_spark-impala_output.py:
Main run file to start creating a model on train_test data + a output on new data without the possibility of testing accuracy.
  
<br>
<br>

# Mailmerge

This script moves data from Impala/HIVE to document through mail_merge and finally writes it to HDFS.
The reason for using for loops is to combine a document with a fixed length and form with multiple other documents which match the same keys in a different database. Hereby moving past the limitations of the normal mail_merge usage, where it is not possible to combine one document with several rows from another source.
