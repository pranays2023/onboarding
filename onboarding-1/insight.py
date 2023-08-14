from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, avg
from pyspark.sql.types import FloatType, DoubleType, IntegerType
import sys

def create_dataframe_from_csv(file_path):
    spark = SparkSession.builder \
        .appName("CSV to DataFrame") \
        .getOrCreate()

    try:
        df = spark.read.csv(file_path, header=True, inferSchema=True)
        df.cache()  # Cache the DataFrame for improved performance
        return df
    except Exception as e:
        print(f"Error reading the CSV file. Details: {e}")
        return None

def analyze_dataframe(df):
    # Print Schema
    print("\nSchema:")
    df.printSchema()
    
    # Number of Columns and Rows
    num_columns = len(df.columns)
    num_rows = df.count()
    print(f"\nNumber of Columns: {num_columns}")
    print(f"Number of Rows: {num_rows}")

    # Check for Nulls
    null_counts = df.select([sum(col(c).isNull().cast("int")).alias(c) for c in df.columns]).collect()[0].asDict()
    print("\nNull Counts:")
    for k, v in null_counts.items():
        print(f"{k}: {v}")

    # Check for Duplicates
    duplicates_count = num_rows - df.dropDuplicates().count()
    print(f"\nNumber of Duplicate Rows: {duplicates_count}")

    # Categorical and Numerical Columns
    numerical_columns = [f.name for f in df.schema.fields if isinstance(f.dataType, (FloatType, DoubleType, IntegerType))]
    categorical_columns = list(set(df.columns) - set(numerical_columns))
    
    print(f"\nNumerical Columns ({len(numerical_columns)}): {numerical_columns}")
    print(f"Categorical Columns ({len(categorical_columns)}): {categorical_columns}")

    # Mean, Median, Mode for Numerical Columns
    if numerical_columns:
        print("\nStatistics for Numerical Columns:")
        for col_name in numerical_columns:
            # Mean
            mean_val = df.select(avg(col(col_name))).collect()[0][0]
            
            # Median (approximate median due to the distributed nature of Spark)
            approx_median = df.stat.approxQuantile(col_name, [0.5], 0.001)[0]
            
            # Mode (using Spark's aggregation)
            mode_val = df.groupBy(col_name).count().orderBy('count', ascending=False).limit(1).collect()[0][col_name]
            
            print(f"\n{col_name} - Mean: {mean_val}, Median: {approx_median}, Mode: {mode_val}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = input("Enter the path to the CSV file: ")
    
    df = create_dataframe_from_csv(file_path)
    
    if df:
        df.show()
        analyze_dataframe(df)
