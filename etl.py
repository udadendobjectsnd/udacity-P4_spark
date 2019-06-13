import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format


config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID']=config.get('AWS', 'AWS_ACCESS_KEY_ID')
os.environ['AWS_SECRET_ACCESS_KEY']=config.get('AWS', 'AWS_SECRET_ACCESS_KEY')


def create_spark_session():
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    '''
    Function to read 'song_data' from S3 or local directory, extract 'songs' and 'artists' tables and load them as parquet files in S3. Each table has its own directory on S3.
    Parameters:
        - spark: Spark session object.
        - input_data: Path to base input data, can be local or S3.
        - output_data: Path to base output data at S3.
    Outputs:
        None
    '''
    # get filepath to song data file
    song_data = input_data + 'song_data/*/*/*/*.json' 
    
    # read song data file
    df = spark.read.json(song_data)

    # extract columns to create songs table
    songs_table =  df.select(df.song_id.alias('song_id'), \
                            df.title.alias('title'), \
                            df.artist_id.alias('artist_id'), \
                            df.year.alias('year'), \
                            df.duration.alias('duration')).dropDuplicates()
    
    # write songs table to parquet files partitioned by year and artist
    songs_table.write.partitionBy(['year', 'artist_id']).mode('append').parquet(output_data + '/songs/')

    # extract columns to create artists table
    artists_table = df.select(df.artist_id.alias('artist_id'), \
                              df.artist_name.alias('artist_name'), \
                              df.artist_location.alias('artist_location'), \
                              df.artist_latitude.alias('artist_latitude'), \
                              df.artist_longitude.alias('artist_longitude')).dropDuplicates()
    
    # write artists table to parquet files
    artists_table.write.mode('append').parquet(output_data + '/artists/')


def process_log_data(spark, input_data, output_data):
    '''
    Function to read 'log-data' from S3 or local directory, extract 'users', 'time' and 'songplays' tables and load them as parquet files in S3. Each table has its own directory on S3.
    Parameters:
        - spark: Spark session object.
        - input_data: Path to base input data, can be local or S3.
        - output_data: Path to base output data at S3.
    Outputs:
        None
    '''
    # get filepath to log data file
    log_data = input_data + 'log-data'

    # read log data file
    df = spark.read.json(log_data) 
    
    # filter by actions for song plays
    df =  df[df['page'] == 'NextSong']

    # extract columns for users table    
    users_table = df.select(df.userId.alias('user_id'), \
                            df.firstName.alias('first_name'), \
                            df.lastName.alias('last_name'), \
                            df.gender.alias('gender'), \
                            df.level.alias('level')).dropDuplicates()
    
    # write users table to parquet files
    users_table.write.mode('append').parquet(output_data + '/users/')

    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda d: datetime.fromtimestamp(d/1000.0), TimestampType())
    df = df.withColumn('timestamp', get_timestamp('ts'))
    
    # create datetime column from original timestamp column
    get_datetime = udf(lambda d: datetime.fromtimestamp(d/1000.0), DateType())
    df = df.withColumn('date', get_datetime('ts'))
    
    # extract columns to create time table
    time_table = df.select(df.ts.alias('start_time'), \
                           hour(df.timestamp).alias('hour'), \
                           dayofmonth(df.timestamp).alias('day'), \
                           weekofyear(df.timestamp).alias('week'), \
                           month(df.timestamp).alias('month'), \
                           year(df.timestamp).alias('year'), \
                           dayofweek(df.timestamp).alias('weekday')).dropDuplicates()
    
    # write time table to parquet files partitioned by year and month
    time_table.write.partitionBy(['year', 'month']).mode('append').parquet(output_data + '/time/')

    # read in song data to use for songplays table
    song_df = spark.read.parquet(output_data + '/songs/')
    # extract columns from joined song and log datasets to create songplays table 
    # songplays_table: songplay_id, start_time, user_id, level, song_id, artist_id, session_id, location, user_agent
    songplays_table_join = df.join(song_df, (df.song == song_df.title) & \
                                            (df.length == song_df.duration)).dropDuplicates()

    songplays_table = songplays_table_join.select(songplays_table_join.ts.alias('start_time'), \
                                                  songplays_table_join.userId.alias('user_id'), \
                                                  songplays_table_join.level.alias('level'), \
                                                  songplays_table_join.song_id.alias('song_id'), \
                                                  songplays_table_join.artist_id.alias('artist_id'), \
                                                  songplays_table_join.sessionId.alias('session_id'), \
                                                  songplays_table_join.location.alias('location'), \
                                                  songplays_table_join.userAgent.alias('user_agent'), \
                                                  year(songplays_table_join.timestamp).alias('year'), \
                                                  month(songplays_table_join.timestamp).alias('month')).dropDuplicates()

    songplays_table.withColumn('songplay_id', monotonically_increasing_id()) 

    # write songplays table to parquet files partitioned by year and month
    songplays_table.write.partitionBy(['year', 'month']).mode('append').parquet(output_data + '/songplays/')
    
    
def main():
    '''
    Main function. Get 'input_data' and 'output_data' from  configuration file 'dl.cfg' and call the functions create_spark_session, process_song_data and process_log_data.
    '''
    spark = create_spark_session()
    input_data = config.get('IO', 'INPUT_DATA')
    output_data = config.get('IO', 'OUTPUT_DATA')
    
    process_song_data(spark, input_data, output_data)    
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()
