#!/usr/bin/env python3

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, TimestampType
import logging
import time
import glob
import psutil
import os
from pyspark.sql.window import Window

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('federato/spark_analysis.log')
    ]
)
logger = logging.getLogger(__name__)

def log_memory_usage():
    """Log current memory usage."""
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024  # in MB
    logger.info(f"Current memory usage: {mem:.2f} MB")

class Timer:
    """Context manager for timing code blocks."""
    def __init__(self, description: str):
        self.description = description
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.time()
        logger.info(f"Starting: {self.description}")
        log_memory_usage()
        return self

    def __exit__(self, *args):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        logger.info(f"Completed: {self.description} in {duration:.2f} seconds")
        log_memory_usage()
        return False

def create_spark_session():
    """Create and configure Spark session."""
    available_memory = psutil.virtual_memory().available
    executor_memory = int(available_memory * 0.8 / 1024 / 1024)  # 80% of available memory in MB
    
    return (SparkSession.builder
            .appName("UserAnalytics")
            .config("spark.executor.memory", f"{executor_memory}m")
            .config("spark.driver.memory", f"{executor_memory}m")
            .config("spark.sql.shuffle.partitions", os.cpu_count() * 4)
            .config("spark.default.parallelism", os.cpu_count() * 4)
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
            .config("spark.sql.adaptive.localShuffleReader.enabled", "true")
            # Fix for Java security manager issue
            .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow")
            .config("spark.executor.extraJavaOptions", "-Djava.security.manager=allow")
            # Local mode settings
            .master("local[*]")
            .config("spark.driver.host", "localhost")
            .config("spark.driver.bindAddress", "localhost")
            .getOrCreate())

def get_schema():
    """Define the schema for CSV files."""
    return StructType([
        StructField("user_id", StringType(), True),
        StructField("session_id", StringType(), True),
        StructField("event_type", StringType(), True),
        StructField("client_event_time", TimestampType(), True),
        StructField("country", StringType(), True),
        StructField("region", StringType(), True),
        StructField("city", StringType(), True),
        StructField("device_family", StringType(), True),
        StructField("os_name", StringType(), True)
    ])

def explore_dataset(df):
    """Explore and log dataset characteristics."""
    with Timer("Dataset Exploration"):
        # Basic counts
        logger.info("\n=== Dataset Overview ===")
        total_rows = df.count()
        logger.info(f"Total rows: {total_rows:,}")
        
        # Column statistics
        for col in df.columns:
            null_count = df.filter(F.col(col).isNull()).count()
            distinct_count = df.select(col).distinct().count()
            null_percentage = (null_count / total_rows) * 100
            logger.info(f"\nColumn: {col}")
            logger.info(f"- Distinct values: {distinct_count:,}")
            logger.info(f"- Null values: {null_count:,} ({null_percentage:.2f}%)")
            
            if distinct_count < 100:  # Only show distribution for categorical columns
                logger.info("- Value distribution:")
                df.groupBy(col).count().orderBy(F.desc("count")).show(5, truncate=False)

def analyze_temporal_patterns(df):
    """Analyze temporal patterns in the data."""
    with Timer("Temporal Analysis"):
        logger.info("\n=== Temporal Patterns ===")
        
        # Add time-based columns
        df = df.withColumn("hour", F.hour("client_event_time"))
        df = df.withColumn("day", F.dayofweek("client_event_time"))
        df = df.withColumn("month", F.month("client_event_time"))
        
        # Hourly patterns
        logger.info("\nHourly Event Distribution:")
        df.groupBy("hour") \
          .agg(F.count("*").alias("events"),
               F.countDistinct("session_id").alias("unique_sessions")) \
          .orderBy("hour") \
          .show(24)
        
        # Daily patterns
        logger.info("\nDaily Event Distribution:")
        df.groupBy("day") \
          .agg(F.count("*").alias("events"),
               F.countDistinct("session_id").alias("unique_sessions")) \
          .orderBy("day") \
          .show()

def analyze_user_behavior(df):
    """Analyze user behavior patterns."""
    with Timer("User Behavior Analysis"):
        logger.info("\n=== User Behavior Analysis ===")
        
        # Session statistics
        session_stats = df.groupBy("session_id").agg(
            F.count("*").alias("events_per_session"),
            F.countDistinct("event_type").alias("unique_events"),
            F.min("client_event_time").alias("session_start"),
            F.max("client_event_time").alias("session_end")
        )
        
        session_stats = session_stats.withColumn(
            "session_duration_minutes",
            F.round(F.unix_timestamp("session_end") - F.unix_timestamp("session_start")) / 60
        )
        
        logger.info("\nSession Duration Statistics:")
        session_stats.select("session_duration_minutes") \
                    .summary("count", "min", "25%", "75%", "max") \
                    .show()
        
        logger.info("\nEvents per Session Statistics:")
        session_stats.select("events_per_session") \
                    .summary("count", "min", "25%", "75%", "max") \
                    .show()

def analyze_event_sequences(df):
    """Analyze event sequences and transitions."""
    with Timer("Event Sequence Analysis"):
        logger.info("\n=== Event Sequence Analysis ===")
        
        # Create window for next event
        window_spec = Window.partitionBy("session_id").orderBy("client_event_time")
        
        df_with_next = df.withColumn(
            "next_event",
            F.lead("event_type").over(window_spec)
        )
        
        # Analyze event transitions
        logger.info("\nTop Event Transitions:")
        df_with_next.filter(F.col("next_event").isNotNull()) \
                    .groupBy("event_type", "next_event") \
                    .count() \
                    .orderBy(F.desc("count")) \
                    .show(10, truncate=False)
        
        # First events in sessions
        logger.info("\nMost Common Session Start Events:")
        df.withColumn(
            "is_first_event",
            F.row_number().over(Window.partitionBy("session_id").orderBy("client_event_time")) == 1
        ).filter("is_first_event") \
         .groupBy("event_type") \
         .count() \
         .orderBy(F.desc("count")) \
         .show(5)

def analyze_geographic_patterns(df):
    """Analyze geographic distribution and patterns."""
    with Timer("Geographic Analysis"):
        logger.info("\n=== Geographic Analysis ===")
        
        # Country distribution
        logger.info("\nEvents by Country:")
        df.groupBy("country") \
          .agg(F.count("*").alias("events"),
               F.countDistinct("session_id").alias("sessions"),
               F.countDistinct("user_id").alias("users")) \
          .orderBy(F.desc("events")) \
          .show(10)
        
        # Region distribution for top countries
        logger.info("\nTop Regions by Country:")
        df.groupBy("country", "region") \
          .count() \
          .orderBy(F.desc("count")) \
          .show(10)

def analyze_device_usage(df):
    """Analyze device and platform usage patterns."""
    with Timer("Device Usage Analysis"):
        logger.info("\n=== Device Usage Analysis ===")
        
        # Device family distribution
        logger.info("\nDevice Family Distribution:")
        df.groupBy("device_family") \
          .agg(F.count("*").alias("events"),
               F.countDistinct("session_id").alias("sessions")) \
          .orderBy(F.desc("events")) \
          .show()
        
        # OS distribution
        logger.info("\nOS Distribution:")
        df.groupBy("os_name") \
          .agg(F.count("*").alias("events"),
               F.countDistinct("session_id").alias("sessions")) \
          .orderBy(F.desc("events")) \
          .show()

def main():
    """Main function to orchestrate the analysis."""
    total_start_time = time.time()
    logger.info("Starting Spark analysis...")
    spark = None
    
    try:
        # Create Spark session
        spark = create_spark_session()
        
        # Load data
        with Timer("Data Loading"):
            csv_pattern = 'federato/2024/*_csv/*.csv'
            csv_files = glob.glob(csv_pattern)
            
            if not csv_files:
                logger.error(f"No CSV files found matching pattern: {csv_pattern}")
                return
            
            logger.info(f"Found {len(csv_files)} CSV files to process")
            
            # Read CSV files with schema
            df = (spark.read.format("csv")
                 .option("header", "true")
                 .schema(get_schema())
                 .load(csv_pattern))
            
            # Early filtering
            df = (df.filter(~F.col("device_family").isin(["iOS", "Android"]))
                 .filter(F.col("user_id").isNotNull())
                 .cache())
        
        # Run analyses
        explore_dataset(df)
        analyze_temporal_patterns(df)
        analyze_user_behavior(df)
        analyze_event_sequences(df)
        analyze_geographic_patterns(df)
        analyze_device_usage(df)
        
        # Log overall execution time
        total_time = time.time() - total_start_time
        logger.info(f"\nTotal analysis completed in {total_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}")
        raise
    finally:
        # Log final memory usage
        log_memory_usage()
        # Stop Spark session
        if spark:
            spark.stop()

if __name__ == "__main__":
    main() 