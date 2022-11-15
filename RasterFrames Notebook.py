# Databricks notebook source
# MAGIC %md # RasterFrames - Landsat Multiband Read Example
# MAGIC 
# MAGIC * Run this on DBR 6.0+ for python 3.6+ support with fstrings.
# MAGIC 
# MAGIC __[Install Cluster Libraries](https://docs.databricks.com/libraries.html#install-a-library-on-a-cluster):__
# MAGIC 
# MAGIC * Download and Install [RasterFrames Assembly JAR](https://github.com/locationtech/rasterframes/releases/download/0.8.4/pyrasterframes-assembly-0.8.4.jar) via [DBFS Cluster Libraries](https://docs.databricks.com/libraries.html#install-a-library-on-a-cluster) to `dbfs:/lib/rasterframes/pyrasterframes-assembly-0.8.4.jar` (see cell #6)
# MAGIC * PyRasterFrames PyPI Coordinates via [Repo Cluster Libraries](https://docs.databricks.com/libraries.html#pypi-package): `pyrasterframes==0.8.4`
# MAGIC * Shapely PyPI Coordinates: `shapely`
# MAGIC * (Optional) Fiona PyPI Coordinates: `fiona`
# MAGIC * (Optional) GeoPandas PyPI Coordinates: `geopandas`
# MAGIC 
# MAGIC __Install GDAL Native Library on the cluster via [Init Script](https://docs.databricks.com/clusters/init-scripts.html#cluster-scoped-init-scripts) (see cell #3):__
# MAGIC 
# MAGIC ```
# MAGIC dbutils.fs.put("/databricks/scripts/gdal_install.sh","""
# MAGIC #!/bin/bash
# MAGIC sudo add-apt-repository ppa:ubuntugis/ppa
# MAGIC sudo apt-get update
# MAGIC sudo apt-get install -y cmake gdal-bin libgdal-dev python3-gdal
# MAGIC """,
# MAGIC True)
# MAGIC ```
# MAGIC 
# MAGIC __Add the following Spark Configurations to your session to improve serialization performance (see cell #12).__
# MAGIC 
# MAGIC * spark.serializer=org.apache.spark.serializer.KryoSerializer 
# MAGIC * spark.kryo.registrator=org.locationtech.rasterframes.util.RFKryoRegistrator 
# MAGIC * spark.kryoserializer.buffer.max=500m

# COMMAND ----------

# MAGIC %md RasterFrames provides a DataFrame-centric view over arbitrary geospatial raster data, enabling spatiotemporal queries, map algebra raster operations, and interoperability with Spark ML. By using the DataFrame as the core cognitive and compute data model, RasterFrames is able to deliver an extensive set of functionality in a form that is both horizontally scalable as well as familiar to general analysts and data scientists. It provides APIs for Python, SQL, and Scala.
# MAGIC 
# MAGIC Through its custom [Spark DataSource](https://rasterframes.io/raster-read.html), RasterFrames can read various raster formats -- including GeoTIFF, JP2000, MRF, and HDF -- and from an [array of services](https://rasterframes.io/raster-read.html#uri-formats), such as HTTP, FTP, HDFS, S3 and WASB. It also supports reading the vector formats GeoJSON and WKT/WKB. RasterFrame contents can be filtered, transformed, summarized, resampled, and rasterized through [200+ raster and vector functions](https://rasterframes.io/reference.html).
# MAGIC 
# MAGIC As part of the LocationTech family of projects, RasterFrames builds upon the strong foundations provided by GeoMesa (spatial operations) , GeoTrellis (raster operations), JTS (geometry modeling) and SFCurve (spatiotemporal indexing), integrating various aspects of these projects into a unified, DataFrame-centric analytics package.
# MAGIC 
# MAGIC For the purposes of demonstrating ingestion, the following Python example reads [two bands](https://en.wikipedia.org/wiki/Multispectral_image) of Landsat 8 imagery (red and near-infrared) over NYC, and combines them into a form commonly used for assessing plant health (NDVI).
# MAGIC 
# MAGIC When `pyrasterframes` is imported or the `create_rf_spark_session()` convenience function is called, RasterFrames registers a User Defined Type (UDT) named `Tile` along with a host of columnar Spark functions implemented as [optimizable Catalyst Expressions](https://databricks.com/blog/2015/04/13/deep-dive-into-spark-sqls-catalyst-optimizer.html), User Defined Functions (UDF), and User Defined Aggregate Functions (UDAF). An `rf_ipython` module is provided to render RasterFrame contents in a visually useful form, such as the table below, where the `red`, `NIR` and `NDVI` Tile columns are rendered with color ramps.

# COMMAND ----------

# MAGIC %md ## Setup GDAL [Init Script](https://docs.databricks.com/clusters/init-scripts.html#cluster-scoped-init-scripts)
# MAGIC 
# MAGIC __1x Add the DBFS path `dbfs:/databricks/scripts/gdal_install.sh` to the cluster init scripts and restart the cluster after running this the first time.__

# COMMAND ----------

# --- Run 1x to setup the init script. ---
# Restart cluster after running.
dbutils.fs.put("/databricks/scripts/gdal_install.sh","""
#!/bin/bash
sudo add-apt-repository ppa:ubuntugis/ppa
sudo apt-get update
sudo apt-get install -y cmake gdal-bin libgdal-dev python3-gdal""",
True)

# COMMAND ----------

# MAGIC %fs ls /databricks/scripts

# COMMAND ----------

# MAGIC %md ## Install RasterFrames on the Cluster

# COMMAND ----------

# MAGIC %md __1x Download the RasterFrames Assembly JAR to `dbfs:/lib/rasterframes/pyrasterframes-assembly-0.8.4.jar`.__

# COMMAND ----------

# MAGIC %sh wget -P /dbfs/lib/rasterframes/ https://github.com/locationtech/rasterframes/releases/download/0.8.4/pyrasterframes-assembly-0.8.4.jar

# COMMAND ----------

# MAGIC %md __Now, You need to install `dbfs:/lib/rasterframes/pyrasterframes-assembly-0.8.4.jar` on the Cluster using [Library UI](https://docs.databricks.com/libraries.html#reference-an-uploaded-jar-python-egg-or-python-wheel).__
# MAGIC 
# MAGIC _Also, you need to install PyPI coordinates for `pyrasterframes==0.8.4` on the cluster, see [here](https://docs.databricks.com/libraries.html#pypi-package)._

# COMMAND ----------

# MAGIC %fs ls /lib/rasterframes/pyrasterframes-assembly-0.8.4.jar

# COMMAND ----------

# MAGIC %md ## Import PyRasterFrames and assorted support functions

# COMMAND ----------

# MAGIC %md _Optionally: Set [Spark configs](https://spark.apache.org/docs/latest/tuning.html#data-serialization) as recommended by Rasterframes [here](https://rasterframes.io/getting-started.html)._
# MAGIC 
# MAGIC `spark.conf.set(...)` and `sqlContext.setConf(...)`can be done in context.

# COMMAND ----------

# MAGIC %scala
# MAGIC 
# MAGIC spark.conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
# MAGIC spark.conf.set("spark.kryo.registrator", "org.locationtech.rasterframes.util.RFKryoRegistrator") 
# MAGIC spark.conf.set("spark.kryoserializer.buffer.max","500m")

# COMMAND ----------

from pyrasterframes import rf_ipython
from pyrasterframes.utils import create_rf_spark_session
from pyspark.sql.functions import lit 
from pyrasterframes.rasterfunctions import *

# Use the provided convenience function to create a basic local SparkContext
spark = create_rf_spark_session()

# COMMAND ----------

# MAGIC %md ## Run the Landsat Multiband Read Analytic

# COMMAND ----------

# Construct a CSV "catalog" for RasterFrames `raster` reader. Catalogs can also be Spark or Pandas DataFrames.
bands = [f'B{b}' for b in [4, 5]]
uris = [f'https://landsat-pds.s3.us-west-2.amazonaws.com/c1/L8/014/032/LC08_L1TP_014032_20190720_20190731_01_T1/LC08_L1TP_014032_20190720_20190731_01_T1_{b}.TIF' for b in bands]
catalog = ','.join(bands) + '\n' + ','.join(uris)

# Read red and NIR bands from Landsat 8 dataset over NYC
rf = spark.read.raster(catalog, bands) \
    .withColumnRenamed('B4', 'red').withColumnRenamed('B5', 'NIR') \
    .withColumn('longitude_latitude', st_reproject(st_centroid(rf_geometry('red')), rf_crs('red'), lit('EPSG:4326'))) \
    .withColumn('NDVI', rf_normalized_difference('NIR', 'red')) \
    .where(rf_tile_sum('NDVI') > 10000)

# COMMAND ----------

results = rf.select('longitude_latitude', rf_tile('red'), rf_tile('NIR'), rf_tile('NDVI'))
displayHTML(rf_ipython.spark_df_to_html(results))
