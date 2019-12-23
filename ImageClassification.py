import tensorflow as tf
import tensorflow_datasets as tfds


dataset_name = 'food101'

dataset = tfds.load(name=dataset_name, split=tfds.Split.TRAIN)
dataset = dataset.shuffle(1024).batch(batch_size)


dropout_rate = 0.5

def conv2d(inputs, filters, stride_size):
    out = tf.nn.conv2d( inputs , filters , strides=[ 1 , stride_size , stride_size , 1 ] , padding=True )
    return tf.nn.relu(out)


def maxpool( inputs , pool_size , stride_size ):
    return tf.nn.max_pool2d( inputs , ksize=[ 1 , pool_size , pool_size , 1 ] , padding='VALID' , strides=[ 1 , stride_size , stride_size , 1 ] )


def dense(inputs, weights):
    x = tf.nn.relu( tf.matmul(inputs, weights))
    return tf.nn.dropout(x, rate=dropout_rate)


initializer = tf.initializers.glorot_uniform()
def get_weight( shape , name ):
    return tf.Variable( initializer( shape ) , name=name , trainable=True , dtype=tf.float32 )

shapes = [
    [ 3 , 3 , 3 , 16 ] ,
    [ 3 , 3 , 16 , 16 ] ,
    [ 3 , 3 , 16 , 32 ] ,
    [ 3 , 3 , 32 , 32 ] ,
    [ 3 , 3 , 32 , 64 ] ,
    [ 3 , 3 , 64 , 64 ] ,
    [ 3 , 3 , 64 , 128 ] ,
    [ 3 , 3 , 128 , 128 ] ,
    [ 3 , 3 , 128 , 256 ] ,
    [ 3 , 3 , 256 , 256 ] ,
    [ 3 , 3 , 256 , 512 ] ,
    [ 3 , 3 , 512 , 512 ] ,
    [ 8192 , 3600 ] ,
    [ 3600 , 2400 ] ,
    [ 2400 , 1600 ] ,
    [ 1600 , 800 ] ,
    [ 800 , 64 ] ,
    [64 , output_classes] ,]

weights = []
for i in range( len( shapes ) ):
    weights.append( get_weight( shapes[ i ] , 'weight{}'.format( i ) ) )

