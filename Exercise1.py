# Mintu Vipin Joy
# 301135483

# 2. Load the data 

df_mintu = spark.read.load("/home/centos/data/sample_libsvm_data.txt", format="libsvm", sep=":", inferSchema="true", header="true")

df_mintu.show()


# 3. Basic investigations

# Count the number of records
df_mintu.count()

# Count the number of columns 
len(df_mintu.columns)

# print((df_mintu.count(), len(df_mintu.columns)))

# Print the inferred schema 
df_mintu.printSchema()


# 4. Use the StringIndexer to index labels

from pyspark.ml.feature import StringIndexer

labelIndexer_mintu = StringIndexer().setInputCol("label").setOutputCol("indexedLabel_mintu")

model_label = labelIndexer_mintu.fit(df_mintu)


# 5. Use the VectorIndexer to index categorical features

from pyspark.ml.feature import VectorIndexer

featureIndexer_mintu = VectorIndexer(maxCategories=4, inputCol="features", outputCol= "indexedFeatures_mintu")

model_feature = featureIndexer_mintu.fit(df_mintu)


# 6. Printout the following:

# a. Name of input column & output column of model_label
model_label.getInputCol()

model_label.getOutputCol()

 
#b.	Name of input & output column of model_feature
model_feature.getInputCol()

model_feature.getOutputCol()

#c.	# of features
model_feature.numFeatures

#d.	Map of categories
model_feature.categoryMaps


# 7. Split your original data into 65% for training and 35% for testing 

training_mintu, testing_mintu  = df_mintu.randomSplit([0.65, 0.35]) 

training_mintu.head()


# 8.Create an estimator object that contains a decision tree classifier

from pyspark.ml.classification import DecisionTreeClassifier

DT_mintu = DecisionTreeClassifier(labelCol="indexedLabel_mintu", featuresCol="indexedFeatures_mintu")


# 9. Create a pipeline object with three stages 

from pyspark.ml import Pipeline

pipeline_mintu = Pipeline(stages=[model_label, model_feature, DT_mintu])


# 10. Fit the training data to the pipeline

model_mintu = pipeline_mintu.fit(training_mintu)


# 11. Predict the testing data

predictions_mintu = model_mintu.transform(testing_mintu)


# 12. Print the schema of the predictions

predictions_mintu.printSchema()


# 13. Print the accuracy of your model and the test error

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel_mintu", predictionCol="prediction")
accuracy = evaluator.evaluate(predictions_mintu)
print ("Test Error = %g" % (1.0 - accuracy))
print("Accuracy of model = %g" % accuracy)


# 14. Show the first 10 predictions with the actual labels and features

predictions_mintu.select("prediction", "indexedLabel_mintu", "features").show(10)
