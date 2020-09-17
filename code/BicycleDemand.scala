package spark_examples

import org.apache.spark.rdd.RDD

object BicycleDemand extends App {
  
  import common._
  import practice._
  import org.apache.spark.sql.Row
  import org.apache.spark.sql.Column
  import org.apache.spark.sql.types.IntegerType
  import org.apache.spark.sql.functions._
  import org.apache.spark.sql.DataFrame
  import org.apache.spark.sql._
  import org.apache.log4j.Logger
  import org.apache.log4j.Level
  import scala.util.{Try,Success,Failure}
  
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  
  val spSession = SparkCommonUtils.spSession
  val spContext = SparkCommonUtils.spContext
  val datadir = SparkCommonUtils.datadir
  
  import org.apache.spark.ml.evaluation.RegressionEvaluator
  import org.apache.spark.mllib.evaluation.RegressionMetrics
  import org.apache.spark.ml.regression.LinearRegression
  import org.apache.spark.ml.{Pipeline,PipelineModel,PredictionModel}
  
  import org.apache.spark.ml.tuning.{ParamGridBuilder,TrainValidationSplit}
  import org.apache.log4j._
  import org.apache.spark.ml.feature.{VectorAssembler,VectorIndexer,Normalizer}
  
  Logger.getLogger("org").setLevel(Level.ERROR)
 
  val df_data = spSession.read.option("header","true").option("inferSchema","true").format("csv").load(datadir + "train.csv")
    
  println(df_data.first())
  
  println("Check the dataset has missing values")
  df_data.select(df_data.columns.map(c => sum(col(c).isNull.cast("int")).alias(c)): _*).show
 
  
  val df = df_data.drop("casual").drop("registered")
  
  println ("Split datetime into meaning columns such as Day, Month, Year, Hour")
  val df1 = df.withColumn("Timestamp",unix_timestamp (col("datetime"),"dd-MM-yyyy HH:mm").cast("timestamp"))
    .withColumn("Date",to_date (col("Timestamp")))
    .withColumn("Day",date_format (col("Date"),"dd").cast("int"))
    .withColumn("Month",date_format (col("Date"),"MM").cast("int"))
    .withColumn("Year",date_format (col("Date"),"yyyy").cast("int"))
    .withColumn("Hour",date_format (col("Timestamp"),"HH").cast("int")) 
    .withColumn("season",df("season").cast("int"))
    .withColumn("holiday", df("holiday").cast("int"))
    .withColumn("workingday", df("workingday").cast("int"))
    .withColumn("weather",df("weather").cast("int"))
    .withColumn("temp", df("temp").cast("double"))
    .withColumn("atemp", df("atemp").cast("double"))
    .withColumn("humidity", df("humidity").cast("double"))
    .withColumn("windspeed", df("windspeed").cast("double"))
    .withColumn("count", df("count").cast("double"))
   
    
   val df2 = df1.drop("Date").drop("datetime").drop("Timestamp")
   
    
   val featureCols = Array ("Day","Month","Year","Hour","season","holiday","workingday","weather","temp","atemp","humidity","windspeed","count")
  
  
   val df3 = df2.select(col("Day"),col("Month"),col("Year"),col("Hour"),col("season"),col("holiday"),col("workingday"),col("weather"),col("temp"),col("atemp"),col("humidity"),col("windspeed"),col("count"))
   
 
   val vectorAssembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
   
   val dataDF = vectorAssembler.transform(df3)
 
   val dataDF1 = dataDF.withColumnRenamed("count", "label")
 
   println(dataDF1.count()) 
   
   val Array(train, test) = dataDF1.randomSplit(Array(0.9, 0.1))
 
   test.show(10)
   println(test.count)
   println(train.count)
 
 
   val lr = new LinearRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)

    // Fit the model
    val lrModel = lr.fit(train)

    // Print the coefficients and intercept for linear regression
    println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

    // Summarize the model over the training set and print out some metrics
    val trainingSummary = lrModel.summary
    
   val predictions = lrModel.transform(test)

    
    predictions.select("features","label","prediction").show()
    
    println (predictions.count())
    
  //  trainingSummary.predictions.show(20,false)
    println(s"numIterations: ${trainingSummary.totalIterations}")
    println(s"objectiveHistory: ${trainingSummary.objectiveHistory.toList}")
 //   trainingSummary.residuals.show()
    println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")

    println(s"r2: ${trainingSummary.r2}")

}  
