# Databricks notebook source
# MAGIC %sql
# MAGIC select * from powerplant

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select AT as Temperature, PE as Power from PowerPlant

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select PE as Power, V as ExhaustVaccum from PowerPlant

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select PE as Power, AP as Pressure from PowerPlant

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select PE as Power, RH as Humidity from PowerPlant

# COMMAND ----------

# MAGIC %scala
# MAGIC val csv = spark.read.option("inferSchema","true").option("header", "true").csv("/FileStore/tables/Folds5x2_pp-2.csv")
# MAGIC csv.show()

# COMMAND ----------

# MAGIC %scala
# MAGIC 
# MAGIC val splits = csv.randomSplit(Array(0.7, 0.3))
# MAGIC val train = splits(0)
# MAGIC val test = splits(1)
# MAGIC val train_rows = train.count()
# MAGIC val test_rows = test.count()
# MAGIC println("Training Rows: " + train_rows + " Testing Rows: " + test_rows)

# COMMAND ----------

# MAGIC %scala
# MAGIC 
# MAGIC import org.apache.spark.ml.feature.VectorAssembler
# MAGIC 
# MAGIC val assembler = new VectorAssembler().setInputCols(Array("AT", "V", "AP", "RH")).setOutputCol("features")
# MAGIC 
# MAGIC val training = assembler.transform(train).select($"features", $"PE".alias("label"))
# MAGIC 
# MAGIC training.show(false)

# COMMAND ----------

# MAGIC %scala
# MAGIC 
# MAGIC import org.apache.spark.ml.regression.LinearRegression
# MAGIC 
# MAGIC val lr = new LinearRegression().setLabelCol("label").setFeaturesCol("features").setMaxIter(10).setRegParam(0.3)
# MAGIC val model = lr.fit(training)
# MAGIC println("Model Trained!")

# COMMAND ----------

# MAGIC %scala
# MAGIC 
# MAGIC val testing = assembler.transform(test).select($"features", $"PE".alias("trueLabel"))
# MAGIC testing.show(false)

# COMMAND ----------

# MAGIC %scala
# MAGIC 
# MAGIC val prediction = model.transform(testing)
# MAGIC val predicted = prediction.select("features", "prediction", "trueLabel")
# MAGIC predicted.show()

# COMMAND ----------

# MAGIC %scala
# MAGIC 
# MAGIC predicted.createOrReplaceTempView("regressionPredictions")

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT trueLabel, prediction FROM regressionPredictions

# COMMAND ----------

# MAGIC %scala
# MAGIC import org.apache.spark.ml.evaluation.RegressionEvaluator
# MAGIC 
# MAGIC val evaluator = new RegressionEvaluator().setLabelCol("trueLabel").setPredictionCol("prediction").setMetricName("rmse")
# MAGIC val rmse = evaluator.evaluate(prediction)
# MAGIC println("Root Mean Square Error (RMSE): " + (rmse))
