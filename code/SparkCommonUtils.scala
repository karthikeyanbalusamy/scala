package common

object SparkCommonUtils {

  import org.apache.spark.sql.SparkSession

  import org.apache.spark.SparkContext

  import org.apache.spark.SparkConf

  //Directory where the data files for the examples exist.
  // val datadir = "D:\\TRAININGS\\EDUREKA APACHE SPARK TRAINING LATEST\\spark-edureka-examples\\data-files\\"
  //val datadir = "C:\\scala\\scala_code\\khu3c\\spark-edureka-examples\\data-files\\"
  val datadir = "C:\\Scala_Spark\\Spark_Eclipse\\spark-edureka-examples\\data-files\\"
  
  
  //A name for the spark instance. Can be any string
  val appName = "Bicycle Demand prediction"

  //Pointer / URL to the Spark instance - embedded
  val sparkMasterURL = "local[2]"
  //val sparkMasterURL = "spark://169.254.2.214:7077"

  //Temp dir required for Spark SQL
  val tempDir = "file:///c:/temp/spark-warehouse"

  var spSession: SparkSession = null
  var spContext: SparkContext = null

  //Initialization. Runs when object is created
  {
    //Need to set hadoop.home.dir to avoid errors during startup
    //System.setProperty("hadoop.home.dir", "C:\\winutils");
    System.setProperty("hadoop.home.dir", "C:\\Scala_Spark\\winutils\\");

    //Create spark configuration object
    val conf = new SparkConf()
      .setAppName(appName)
      .setMaster(sparkMasterURL)
      .set("spark.executor.memory", "2g")
      .set("spark.sql.shuffle.partitions", "2")

    //Get or create a spark context. Creates a new instance if not exists	
    spContext = SparkContext.getOrCreate(conf)

    //Create a spark SQL session
    spSession = SparkSession
      .builder()
      .appName(appName)
      .master(sparkMasterURL)
      .config("spark.sql.warehouse.dir", tempDir)
      .getOrCreate()

  }
}