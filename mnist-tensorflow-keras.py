# Databricks notebook source
# MAGIC %md
# MAGIC # Distributed deep learning training using TensorFlow and Keras with HorovodRunner
# MAGIC 
# MAGIC This notebook illustrates the use of HorovodRunner for distributed training with the `tensorflow.keras` API. 
# MAGIC It first shows how to train a model on a single node, and then shows how to adapt the code using HorovodRunner for distributed training. 
# MAGIC The notebook runs on CPU and GPU clusters. 
# MAGIC 
# MAGIC ## Requirements
# MAGIC Databricks Runtime 7.0 ML or above.  
# MAGIC HorovodRunner is designed to improve model training performance on clusters with multiple workers, but multiple workers are not required to run this notebook.

# COMMAND ----------

# MAGIC %md ## Create function to prepare data
# MAGIC 
# MAGIC The `get_dataset()` function prepares the data for training. This function takes in `rank` and `size` arguments so it can be used for both single-node and distributed training. In Horovod, `rank` is a unique process ID and `size` is the total number of processes. 
# MAGIC 
# MAGIC This function downloads the data from `keras.datasets`, distributes the data across the available nodes, and converts the data to shapes and types needed for training.

# COMMAND ----------

def get_dataset(num_classes, rank=0, size=1):
  from tensorflow import keras
  
  (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data('MNIST-data-%d' % rank)
  x_train = x_train[rank::size]
  y_train = y_train[rank::size]
  x_test = x_test[rank::size]
  y_test = y_test[rank::size]
  x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
  x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')
  x_train /= 255
  x_test /= 255
  y_train = keras.utils.to_categorical(y_train, num_classes)
  y_test = keras.utils.to_categorical(y_test, num_classes)
  return (x_train, y_train), (x_test, y_test)

# COMMAND ----------

# MAGIC %md ## Create function to train model
# MAGIC The `get_model()` function defines the model using the `tensorflow.keras` API. This code is adapted from the [Keras MNIST convnet example](https://keras.io/examples/vision/mnist_convnet/). 

# COMMAND ----------

def get_model(num_classes):
  from tensorflow.keras import models
  from tensorflow.keras import layers
  
  model = models.Sequential()
  model.add(layers.Conv2D(32, kernel_size=(3, 3),
                   activation='relu',
                   input_shape=(28, 28, 1)))
  model.add(layers.Conv2D(64, (3, 3), activation='relu'))
  model.add(layers.MaxPooling2D(pool_size=(2, 2)))
  model.add(layers.Dropout(0.25))
  model.add(layers.Flatten())
  model.add(layers.Dense(128, activation='relu'))
  model.add(layers.Dropout(0.5))
  model.add(layers.Dense(num_classes, activation='softmax'))
  return model

# COMMAND ----------

# MAGIC %md ## Run training on single node

# COMMAND ----------

# MAGIC %md The `train()` function in the following cell defines single-node training code with `tensorflow.keras`. 

# COMMAND ----------

# Specify training parameters
batch_size = 128
epochs = 2
num_classes = 10

def train(learning_rate=1.0):
  from tensorflow import keras
  
  (x_train, y_train), (x_test, y_test) = get_dataset(num_classes)
  model = get_model(num_classes)

  # Specify the optimizer (Adadelta in this example), using the learning rate input parameter of the function so that Horovod can adjust the learning rate during training
  optimizer = keras.optimizers.Adadelta(lr=learning_rate)

  model.compile(optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy'])

  model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=2,
            validation_data=(x_test, y_test))
  return model

# COMMAND ----------

# MAGIC %md
# MAGIC Run the `train()` function to train a model on the driver node. The process takes several minutes. The accuracy improves with each epoch. 

# COMMAND ----------

model = train(learning_rate=0.1)

# COMMAND ----------

# MAGIC %md Calculate and print the loss and accuracy of the model.

# COMMAND ----------

_, (x_test, y_test) = get_dataset(num_classes)
loss, accuracy = model.evaluate(x_test, y_test, batch_size=128)
print("loss:", loss)
print("accuracy:", accuracy)

# COMMAND ----------

# MAGIC %md ## Migrate to HorovodRunner for distributed training
# MAGIC 
# MAGIC This section shows how to modify the single-node code to use Horovod. For more information about Horovod, see the [Horovod documentation](https://horovod.readthedocs.io/en/stable/).  
# MAGIC 
# MAGIC First, create a directory to save model checkpoints.

# COMMAND ----------

import os
import time

# Remove any existing checkpoint files
dbutils.fs.rm(("/ml/MNISTDemo/train"), recurse=True)

# Create directory
checkpoint_dir = '/dbfs/ml/MNISTDemo/train/{}/'.format(time.time())
os.makedirs(checkpoint_dir)
print(checkpoint_dir)

# COMMAND ----------

# MAGIC %md The following cell shows how to modify the single-node code of the previously defined `train()` function to take advantage of distributed training.  

# COMMAND ----------

def train_hvd(checkpoint_path, learning_rate=1.0):
  
  # Import tensorflow modules to each worker
  from tensorflow.keras import backend as K
  from tensorflow.keras.models import Sequential
  import tensorflow as tf
  from tensorflow import keras
  import horovod.tensorflow.keras as hvd
  
  # Initialize Horovod
  hvd.init()

  # Pin GPU to be used to process local rank (one GPU per process)
  # These steps are skipped on a CPU cluster
  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
  if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

  # Call the get_dataset function you created, this time with the Horovod rank and size
  (x_train, y_train), (x_test, y_test) = get_dataset(num_classes, hvd.rank(), hvd.size())
  model = get_model(num_classes)

  # Adjust learning rate based on number of GPUs
  optimizer = keras.optimizers.Adadelta(lr=learning_rate * hvd.size())

  # Use the Horovod Distributed Optimizer
  optimizer = hvd.DistributedOptimizer(optimizer)

  model.compile(optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy'])

  # Create a callback to broadcast the initial variable states from rank 0 to all other processes.
  # This is required to ensure consistent initialization of all workers when training is started with random weights or restored from a checkpoint.
  callbacks = [
      hvd.callbacks.BroadcastGlobalVariablesCallback(0),
  ]

  # Save checkpoints only on worker 0 to prevent conflicts between workers
  if hvd.rank() == 0:
      callbacks.append(keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only = True))

  model.fit(x_train, y_train,
            batch_size=batch_size,
            callbacks=callbacks,
            epochs=epochs,
            verbose=2,
            validation_data=(x_test, y_test))

# COMMAND ----------

# MAGIC %md
# MAGIC Now you are ready to use HorovodRunner to distribute the work of training the model. 
# MAGIC 
# MAGIC The HorovodRunner parameter `np` sets the number of processes. This example uses a cluster with two workers, each with a single processor, so set `np=2`. (If you use `np=-1`, HorovodRunner trains using a single process on the driver node.)
# MAGIC 
# MAGIC Under the hood, HorovodRunner takes a Python method that contains deep learning training code with Horovod hooks. HorovodRunner pickles the method on the driver and distributes it to Spark workers. A Horovod MPI job is embedded as a Spark job using the barrier execution mode. The first executor collects the IP addresses of all task executors using BarrierTaskContext and triggers a Horovod job using `mpirun`. Each Python MPI process loads the pickled user program, deserializes it, and runs it.
# MAGIC 
# MAGIC For more information, see [HorovodRunner API documentation](https://databricks.github.io/spark-deep-learning/#api-documentation). 

# COMMAND ----------

from sparkdl import HorovodRunner

checkpoint_path = checkpoint_dir + '/checkpoint-{epoch}.ckpt'
learning_rate = 0.1
hr = HorovodRunner(np=2)
hr.run(train_hvd, checkpoint_path=checkpoint_path, learning_rate=learning_rate)

# COMMAND ----------

# MAGIC %md ## Load the model trained by HorovodRunner for inference
# MAGIC The following code shows how to access and load the model after completing training with HorovodRunner. It uses the TensorFlow method `tf.train.latest_checkpoint()` to access the latest saved checkpoint file.

# COMMAND ----------

import tensorflow as tf

hvd_model = get_model(num_classes)
hvd_model.compile(optimizer=tf.keras.optimizers.Adadelta(lr=learning_rate),
                loss='categorical_crossentropy',
                metrics=['accuracy'])
hvd_model.load_weights(tf.train.latest_checkpoint(os.path.dirname(checkpoint_path)))

# COMMAND ----------

# MAGIC %md Evaluate the model's performance on the test dataset.

# COMMAND ----------

_, (x_test, y_test) = get_dataset(num_classes)
loss, accuracy = hvd_model.evaluate(x_test, y_test, batch_size=128)
print("loaded model loss and accuracy:", loss, accuracy)

# COMMAND ----------

# MAGIC %md Use the model to make predictions on new data. For example purposes, use the first 10 observations in the test dataset to stand in for new data. 

# COMMAND ----------

import numpy as np

# Use rint() to round the predicted values to the nearest integer
preds = np.rint(hvd_model.predict(x_test[0:9]))
preds
