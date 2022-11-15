# Databricks notebook source
# MAGIC %md ## [GeoSpark](https://datasystemslab.github.io/GeoSpark/ ) Shapefile Notebook 
# MAGIC 
# MAGIC * For Shapefile see https://datasystemslab.github.io/GeoSpark/tutorial/rdd/#from-shapefile
# MAGIC * Data from https://data.cityofnewyork.us/Housing-Development/Shapefiles-and-base-map/2k7f-6s2k, exported as Shapefiles.
# MAGIC * Data placed on DBFS at `dbfs:/ml/blogs/geospatial/shapefiles/nyc/nyc_buildings.*`
# MAGIC 
# MAGIC __[Install Cluster Libraries](https://docs.databricks.com/libraries.html#install-a-library-on-a-cluster):__
# MAGIC 
# MAGIC * GeoSpark Maven Coordinates: `org.datasyslab:geospark:1.2.0`
# MAGIC * GeoSpark SQL Maven Coordinates: `org.datasyslab:geospark-sql_2.3:1.2.0`

# COMMAND ----------

# MAGIC %md ## Register GeoSpark with Spark SQL

# COMMAND ----------

# MAGIC %scala
# MAGIC 
# MAGIC import com.vividsolutions.jts.geom.{Coordinate, Geometry, GeometryFactory}
# MAGIC import org.datasyslab.geospark.formatMapper.shapefileParser.ShapefileReader
# MAGIC import org.datasyslab.geospark.spatialRDD.SpatialRDD
# MAGIC import org.datasyslab.geosparksql.utils.{Adapter, GeoSparkSQLRegistrator}
# MAGIC 
# MAGIC GeoSparkSQLRegistrator.registerAll(sqlContext)

# COMMAND ----------

# MAGIC %md ## Create DataFrame from NYC Building shapefiles

# COMMAND ----------

# MAGIC %scala
# MAGIC 
# MAGIC var spatialRDD = new SpatialRDD[Geometry]
# MAGIC spatialRDD = ShapefileReader.readToGeometryRDD(sc, "/ml/blogs/geospatial/shapefiles/nyc")
# MAGIC 
# MAGIC var rawSpatialDf = Adapter.toDf(spatialRDD,spark)
# MAGIC rawSpatialDf.createOrReplaceTempView("rawSpatialDf")
# MAGIC 
# MAGIC //cache to speed up queries
# MAGIC rawSpatialDf.repartition(spark.sparkContext.defaultParallelism).cache.count

# COMMAND ----------

# MAGIC %md ## Display using SQL Syntax
# MAGIC 
# MAGIC __Can use any of the other Spark language bindings (Python, Scala / Java, R, and SQL).__

# COMMAND ----------

# MAGIC %sql 
# MAGIC 
# MAGIC SELECT *,
# MAGIC        ST_GeomFromWKT(geometry) AS geometry -- UDF to convert WKT to Geometry 
# MAGIC FROM   rawspatialdf 

# COMMAND ----------

# MAGIC %md __Display Chart of tallest buildings in NYC.__

# COMMAND ----------

# MAGIC %sql 
# MAGIC 
# MAGIC SELECT name, 
# MAGIC        round(Cast(num_floors AS DOUBLE), 0) AS num_floors --String to Number
# MAGIC FROM   rawspatialdf 
# MAGIC WHERE  name <> ''
# MAGIC ORDER  BY num_floors DESC LIMIT 5
