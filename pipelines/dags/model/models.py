import re
import tensorflow as tf
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras import Model, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.layers import Add,Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Activation, AveragePooling2D,ZeroPadding2D, GlobalAveragePooling2D, AveragePooling2D, GlobalMaxPooling2D, concatenate
from tensorflow.keras import layers

# AlexNet model
class AlexNet(Model):
    def __init__(self, input_shape, num_classes):
        super(AlexNet, self).__init__(name='alexnet')

        self.conv1 = Conv2D(96, kernel_size=(11,11), strides= 4,
                        padding= 'valid', activation= 'relu',
                        input_shape= input_shape,
                        kernel_initializer= 'he_normal')
                        
        self.maxpool1 = MaxPooling2D(pool_size=(3,3), strides= (2,2),
                              padding= 'valid', data_format= None)

        self.conv2 = Conv2D(256, kernel_size=(5,5), strides= 1,
                        padding= 'same', activation= 'relu',
                        kernel_initializer= 'he_normal')
        self.maxpool2 = MaxPooling2D(pool_size=(3,3), strides= (2,2),
                              padding= 'valid', data_format= None)

        self.conv3 = Conv2D(384, kernel_size=(3,3), strides= 1,
                        padding= 'same', activation= 'relu',
                        kernel_initializer= 'he_normal')

        self.conv4 = Conv2D(384, kernel_size=(3,3), strides= 1,
                        padding= 'same', activation= 'relu',
                        kernel_initializer= 'he_normal')

        self.conv5 = Conv2D(256, kernel_size=(3,3), strides= 1,
                        padding= 'same', activation= 'relu',
                        kernel_initializer= 'he_normal')

        self.maxpool3 = MaxPooling2D(pool_size=(3,3), strides= (2,2),
                              padding= 'valid', data_format= None)
    
        self.flatten1 = Flatten()
        self.dropout1 = Dropout(0.5)
        self.fc1 = Dense(4096, activation= 'relu')
        self.dropout2 = Dropout(0.5)
        self.fc2 = Dense(4096, activation= 'relu')
        # self.fc3 = Dense(1000, activation= 'relu')
        self.fc3 = Dense(num_classes, activation= 'softmax')
        
    def call(self, input_tensor, training=False):
        
        x = self.conv1(input_tensor)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = self.maxpool2(x)
    
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.maxpool3(x)
        x = self.flatten1(x)
        x = self.dropout1(x, training=training)
        x = self.fc1(x)
        x = self.dropout2(x, training=training)
        x = self.fc2(x)
        x = self.fc3(x)
        # x = self.fc4(x)
    
        return x
#cnn
class Conv(layers.Layer):
  def __init__(self, filters, kernel_size, input_shape):
        super(Conv, self).__init__(name='cnn_conv_block')

        self.conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, input_shape=input_shape)
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
        
  def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = self.relu(x)
        x = self.pool(x)
        return x

class CNN(Model):
  def __init__(self, input_shape, num_classes):
    super(CNN, self).__init__(name='cnn')
    self.conv1 = Conv(filters=32, kernel_size=(3, 3), input_shape=input_shape)
    self.conv2 = Conv(filters=64, kernel_size=(3, 3), input_shape=input_shape)
    self.conv3 = Conv(filters=128, kernel_size=(3, 3), input_shape=input_shape)
    self.conv4 = Conv(filters=128, kernel_size=(3, 3), input_shape=input_shape)
    self.flatten = Flatten()
    self.fc1 = Dense(512, activation= 'relu')
    self.fc2 = Dense(num_classes, activation= 'relu')
  
  def call(self, input_tensor, training=False):
    x = self.conv1(input_tensor)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.conv4(x)
    x = self.flatten(x)
    x = self.fc1(x)
    x = self.fc2(x)

    return x

#RESNET
class Identity_Block(layers.Layer):
    '''
    Implementation of identity block with bottleneck architecture
    Arguments:
    f -       defines shpae of filter in the middle layer of the main path
    filters - list of integers, defining the number of filters in each layer of the main path
    stage -   defines the block position in the network
    block -   used for naming convention
    '''
    def __init__(self, f, filters, stage, block):
        super(Identity_Block, self).__init__(name='identity_block')
        self.f = f
        self.filters = filters
        self.stage = stage
        self.block = block
        
    def build(self, input_shape):
        # defining base name for block
        conv_base_name = 'res' + str(self.stage) + self.block + '_'
        bn_base_name = 'bn' + str(self.stage) + self.block + '_'

        # retrieve number of filters in each layer of main path
        # NOTE: f3 must be equal to n_C. That way dimensions of the third component will match the dimension of original input to identity block
        f1, f2, f3 = self.filters

        # Batch normalization must be performed on the 'channels' axis for input. It is 3, for our case
        bn_axis = 3
        
        #first component-main path
        self.conv1 = Conv2D(f1, (1, 1), strides = (1,1), padding = 'valid', name = conv_base_name + 'first_component', kernel_initializer = glorot_uniform(seed=0), input_shape=input_shape)
        self.bn1 = BatchNormalization(axis = bn_axis, name = bn_base_name + 'first_component')
        self.activation_fn = Activation('relu')
        
        #second component- main path
        self.conv2 = Conv2D(f2,  kernel_size = (self.f, self.f), strides = (1,1), padding = 'same', name = conv_base_name + 'second_component', kernel_initializer = glorot_uniform(seed=0))
        self.bn2 = BatchNormalization(axis = bn_axis, name = bn_base_name + 'second_component')
        
        #third component-  main path
        self.conv3 = Conv2D(f3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_base_name + 'third_component', kernel_initializer = glorot_uniform(seed=0))
        self.bn3 = BatchNormalization(axis = bn_axis, name = bn_base_name + 'third_component')
        
        # "Addition step" - skip-connection value merges with main path
        # NOTE: both values have same dimensions at this point, so no operation is required to match dimensions
        self.add = Add()

    def call(self, x, training=False): 
        '''
        Arguments:
        X - input tensor to the block of shape (m, n_H_prev, n_W_prev, n_C_prev)
       
        Returns: 
        X - output is a tensor of shape (n_H, n_W, n_C) which matches (m, n_H_prev, n_W_prev, n_C_prev)
        '''
        # save input for "addition" to last layer output; step in skip-connection
        skip_conn = x

        x= self.conv1(x)
        x= self.bn1(x, training=training)
        x= self.activation_fn(x)
        
        x= self.conv2(x)
        x=self.bn2(x, training=training)
        x=self.activation_fn(x)
        
        x=self.conv3(x)
        x=self.bn3(x, training=training)
        
        # "Addition step" - skip-connection value merges with main path
        # NOTE: both values have same dimensions at this point, so no operation is required to match dimensions
        x=self.add([x, skip_conn])
        x=self.activation_fn(x)
        return x
    
class Conv_Block(layers.Layer):
    def __init__(self, f, filters, stage, block, s, input_shape):
        super(Conv_Block, self).__init__(name='conv_block')
        
        # defining base name for block
        conv_base_name = 'res' + str(stage) + block + '_'
        bn_base_name = 'bn' + str(stage) + block + '_'

        # retrieve number of filters in each layer of main path
        # NOTE: f3 must be equal to n_C. That way dimensions of the third component will match the dimension of original input to identity block
        f1, f2, f3 = filters

        # Batch normalization must be performed on the 'channels' axis for input. It is 3, for our case
        bn_axis = 3
        
        #first component-main path
        self.conv1 = Conv2D(f1, (1, 1), strides = (s,s), padding = 'valid', name = conv_base_name + 'first_component', kernel_initializer = glorot_uniform(seed=0), input_shape=input_shape)
        self.bn1 = BatchNormalization(axis = bn_axis, name = bn_base_name + 'first_component')
        self.activation_fn = Activation('relu')
        
        #second component- main path
        self.conv2 = Conv2D(f2,  kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_base_name + 'second_component', kernel_initializer = glorot_uniform(seed=0))
        self.bn2 = BatchNormalization(axis = bn_axis, name = bn_base_name + 'second_component')
        # self.act2 = Activation('relu')
        
        #third component-  main path
        self.conv3 = Conv2D(f3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_base_name + 'third_component', kernel_initializer = glorot_uniform(seed=0))
        self.bn3 = BatchNormalization(axis = bn_axis, name = bn_base_name + 'third_component')
        
        ##### Convolve skip-connection value to match its dimensions to third layer output's dimensions #### 
        self.conv_skip_conn = Conv2D(f3, (1, 1), strides = (s,s), padding = 'valid', name = conv_base_name + 'merge', kernel_initializer = glorot_uniform(seed=0))
        self.bn_skip_conn = BatchNormalization(axis = 3, name = bn_base_name + 'merge')
        
        # "Addition step" 
        #both values have same dimensions at this point
        self.add = Add()

    def call(self, x, training=False):
        # save input for "addition" to last layer output; step in skip-connection
        skip_conn = x

        x= self.conv1(x)
        x= self.bn1(x, training=training)
        x= self.activation_fn(x)
        
        x= self.conv2(x)
        x=self.bn2(x, training=training)
        x=self.activation_fn(x)
        
        x=self.conv3(x)
        x=self.bn3(x, training=training)

        #skip connection
        skip_conn = self.conv_skip_conn(skip_conn)
        skip_conn = self.bn_skip_conn(skip_conn)
            
        # "Addition step" 
        # NOTE: both values have same dimensions at this point
        x = self.add([x, skip_conn])
        x = self.activation_fn(x)

        return x
  
class ResNet(Model):
    def __init__(self, input_shape, num_classes):
        super(ResNet, self).__init__(name='resnet')
        # X = Input(self.input_shapes)
        self.zero_padding = ZeroPadding2D((3, 3), data_format='channels_last', input_shape=input_shape)
        
        # Stage 1
        self.conv = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv_0', kernel_initializer = glorot_uniform(seed=0), input_shape=input_shape)
        self.bn = BatchNormalization(axis = 3, name = 'bn_1')
        self.activation_fn = Activation('relu')
        self.maxpool = MaxPooling2D((3, 3), strides=(2, 2))
        
        #Stage 2
        self.conv_block1 = Conv_Block(f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1, input_shape=input_shape)
        self.identity_block1_1 = Identity_Block(3, [64, 64, 256], stage=2, block='b')
        self.identity_block1_2 = Identity_Block(3, [64, 64, 256], stage=2, block='c')
        
        #Stage 3
        self.conv_block2 = Conv_Block(f=3, filters=[128, 128, 512], stage=3, block='a', s=2, input_shape=input_shape)
        self.identity_block2_1 = Identity_Block(3, [128, 128, 512], stage=3, block='b')
        self.identity_block2_2 = Identity_Block(3, [128, 128, 512], stage=3, block='c')
        self.identity_block2_3 = Identity_Block(3, [128, 128, 512], stage=3, block='d')
        
        # Stage 4
        self.conv_block3 = Conv_Block(f=3, filters=[256, 256, 1024], stage=4, block='a', s=2, input_shape=input_shape)
        self.identity_block3_1 = Identity_Block(3, [256, 256, 1024], stage=4, block='b')
        self.identity_block3_2 = Identity_Block(3, [256, 256, 1024], stage=4, block='c')
        self.identity_block3_3 = Identity_Block(3, [256, 256, 1024], stage=4, block='d')
        self.identity_block3_4 = Identity_Block(3, [256, 256, 1024], stage=4, block='e')
        self.identity_block3_5 = Identity_Block(3, [256, 256, 1024], stage=4, block='f')

        # Stage 5
        self.conv_block4 = Conv_Block(f=3, filters=[512, 512, 2048], stage=5, block='a', s=2, input_shape=input_shape)
        self.identity_block4_1 = Identity_Block(3, [512, 512, 2048], stage=5, block='b')
        self.identity_block4_2 = Identity_Block(3, [512, 512, 2048], stage=5, block='c')
        
        self.avg_pool = AveragePooling2D((2, 2), name='avg_pool')
        self.flatten = Flatten()
        self.dense = Dense(num_classes, activation='softmax', name='fc' + str(num_classes), kernel_initializer = glorot_uniform(seed=0))
        
    def call(self, X, training=False):
        # plug in input_shape to define the input tensor
        # X = Input(self.input_shapes)
        
        # Zero-Padding : pads the input with a pad of (3,3)
        X = self.zero_padding(X)
        
        # Stage 1
        X = self.conv(X)

        X = self.bn(X, training=training)
        X = self.activation_fn(X)
        X = self.maxpool(X)
   
        # Stage 2
        X = self.conv_block1(X,training=training)
        X = self.identity_block1_1(X, training=training)
        X = self.identity_block1_2(X, training=training)

        # Stage 3
        X = self.conv_block2(X,training=training)
        X = self.identity_block2_1(X,training=training)
        X = self.identity_block2_2(X,training=training)
        X = self.identity_block2_3(X,training=training)

        # Stage 4
        X = self.conv_block3(X,training=training)
        X = self.identity_block3_1(X,training=training)
        X = self.identity_block3_2(X,training=training)
        X = self.identity_block3_3(X,training=training)
        X = self.identity_block3_4(X,training=training)
        X = self.identity_block3_5(X,training=training)

        # Stage 5
        X = self.conv_block4(X,training=training)
        X = self.identity_block4_1(X,training=training)
        X = self.identity_block4_1(X,training=training)

        # Average Pooling
        X = self.avg_pool(X)

        # output layer
        X = self.flatten(X)
        X = self.dense(X)
            
        return X
# Inception net

class Conv_2D_Block(layers.Layer):
  def __init__(self, filters, kernel, input_shape, strides=(1, 1), padding="same", activation = "relu"):
        super(Conv_2D_Block, self).__init__(name='conv_2D_block')
        self.conv = Conv2D(filters, kernel, strides=strides, padding=padding, kernel_initializer="he_normal", input_shape=input_shape)
        self.activation = Activation(activation)
        
  def call(self, inputs):
        x = self.conv(inputs)
        x = BatchNormalization()(x)
        x = self.activation(x)
        return x


class Inceptionv1_Module(layers.Layer):
  def __init__(self, filterB1, filterB2_1, filterB2_2, filterB3_1, filterB3_2, filterB4_2, i, input_shape):
        super(Inceptionv1_Module, self).__init__(name=f'Inceptionv1_module_{i}')
        self.branch1 = Conv_2D_Block(filterB1, (1, 1), input_shape, padding='valid')
        self.branch2_1 = Conv_2D_Block(filterB2_1, (1, 1), input_shape, padding='valid')
        self.branch2_2 = Conv_2D_Block(filterB2_2, (3, 3), input_shape)
        self.branch3_1 = Conv_2D_Block(filterB3_1, (1, 1), input_shape, padding='valid')
        self.branch3_2 = Conv_2D_Block(filterB3_2, (5, 5), input_shape)
        self.branch4_1 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')
        self.branch4_2 = Conv_2D_Block(filterB4_2, (1, 1), input_shape) 
        self.stage_no = i

        
  def call(self, inputs):
        branch1 = self.branch1(inputs)
        branch2 = self.branch2_1(inputs)
        branch2 = self.branch2_2(branch2)
        branch3 = self.branch3_1(inputs)
        branch3 = self.branch3_2(branch3)
        branch4 = self.branch4_1(inputs)
        branch4 = self.branch4_2(branch4)

        return concatenate([branch1, branch2, branch3, branch4], axis=-1)


class MLP(layers.Layer):
  def __init__(self, pooling, dropout_rate, problem_type, num_classes):
        super(MLP, self).__init__(name='MLP')
        self.pooling = GlobalAveragePooling2D() if pooling == 'avg' else GlobalMaxPooling2D()
        self.dropout = Dropout(dropout_rate) if dropout_rate else False
        self.output_layer = Dense(num_classes, activation = 'softmax' if problem_type == 'Classification' else 'linear')
        
  def call(self, inputs):
        x = self.pooling(inputs)
        # Final Dense Outputting Layer for the outputs
        x = Flatten()(x)
        if self.dropout:
            x = self.dropout(x)
        outputs = self.output_layer(x)
        return outputs

class Auxilliary_Module(layers.Layer):
  def __init__(self, pooling, dropout_rate, problem_type, num_classes, i, input_shape):
        super(Auxilliary_Module, self).__init__(name=f'Auxilliary_Module_{i}')
        self.avgPooling = AveragePooling2D((5, 5), strides=(3, 3), padding='valid')
        self.conv1x1 = Conv_2D_Block(64, (1, 1), input_shape)
        self.MLP = MLP(pooling, dropout_rate, problem_type, num_classes)

  def call(self, inputs):
        aux_pool = self.avgPooling(inputs)
        aux_conv = self.conv1x1(aux_pool)
        aux_output = self.MLP(aux_conv)
        return aux_output


class InceptionNet(Model):
  def __init__(self, input_shape, num_classes, num_filters, problem_type='Regression', pooling='avg', dropout_rate=False):
        super(InceptionNet, self).__init__(name='InceptionNet')

        self.inputs = tf.keras.layers.Input(input_shape)
        self.stem_conv7x7 = Conv_2D_Block(num_filters, (7, 7), strides=(2, 2))
        self.stem_maxPool_1 = MaxPooling2D((3, 3), strides=(2, 2))
        self.stem_conv3x3 = Conv_2D_Block(num_filters*3, (3, 3))
        self.stem_conv1x1 = Conv_2D_Block(num_filters, (1, 1), strides=(2, 2))
        self.stem_maxPool_2 = MaxPooling2D((3, 3), strides=(2, 2))
        self.inception_block_1 = Inceptionv1_Module(64, 96, 128, 16, 32, 32, 1)
        self.inception_block_2 = Inceptionv1_Module(128, 128, 192, 32, 96, 64, 2)
        self.inception_block_3 = Inceptionv1_Module(192, 96, 208, 16, 48, 64, 3)
        self.inception_block_4 = Inceptionv1_Module(160, 112, 224, 24, 64, 64, 4)
        self.inception_block_5 = Inceptionv1_Module(128, 128, 256, 24, 64, 64, 5)
        self.inception_block_6 = Inceptionv1_Module(112, 144, 288, 32, 64, 64, 6)
        self.inception_block_7 = Inceptionv1_Module(256, 160, 320, 32, 128, 128, 7)
        self.inception_block_8 = Inceptionv1_Module(256, 160, 320, 32, 128, 128, 8)
        self.inception_block_9 = Inceptionv1_Module(384, 192, 384, 48, 128, 128, 9)
        self.maxPool1 = MaxPooling2D((3, 3), strides=(2, 2))
        self.maxPool2 = MaxPooling2D((3, 3), strides=(2, 2))
        self.auxiliary_output_1 = Auxilliary_Module(pooling, dropout_rate, problem_type, num_classes, 1)
        self.auxiliary_output_2 = Auxilliary_Module(pooling, dropout_rate, problem_type, num_classes, 2)
        self.final_output = MLP(self.pooling, self.dropout_rate, self.problem_type, self.num_classes)
        

  def call(self, X, training=False):
        aux_output_1 = None
        aux_output_2 = None
        inputs = self.inputs(X)
        # Stem
        x = self.stem_conv7x7(inputs)
        x = self.stem_maxPool_1(x)
        x = self.stem_conv3x3(x)
        x = self.stem_conv1x1(x)
        x = self.stem_maxPool_2(x)
        # Inception Blocks
        x = self.inception_block_1(x)  # Inception Block 1
        x = self.inception_block_2(x)  # Inception Block 2

        if training:
            aux_output_0 = self.auxiliary_output_1(x)

        x = self.maxPool1(x)
        x = self.inception_block_3(x)  # Inception Block 3
        x = self.inception_block_4(x)  # Inception Block 4
        x = self.inception_block_5(x)  # Inception Block 5
        x = self.inception_block_6(x)  # Inception Block 6
        x = self.inception_block_7(x)  # Inception Block 7

        if training:
            aux_output_1 = self.auxiliary_output_1(x)

        x = self.maxPool2(x)
        x = self.inception_block_8(x)  # Inception Block 8
        x = self.inception_block_9(x)  # Inception Block 9

        # Final Dense MLP Layer for the outputs
        final_output = self.final_output(x)
        # model = Model(inputs, outputs=[final_output, aux_output_0, aux_output_1], name='Inception_v1')
        return final_output, aux_output_0, aux_output_1


# Inception V2

def Conv_2D_Block(x, model_width, kernel, strides=(1, 1), padding="same"):
    # 2D Convolutional Block with BatchNormalization
    x = tf.keras.layers.Conv2D(model_width, kernel, strides=strides, padding=padding, kernel_initializer="he_normal")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    return x


def classifier(inputs, class_number):
    # Construct the Classifier Group
    # inputs       : input vector
    # class_number : number of output classes
    out = tf.keras.layers.Dense(class_number, activation='softmax')(inputs)
    return out


def regressor(inputs, feature_number):
    # Construct the Regressor Group
    # inputs         : input vector
    # feature_number : number of output features
    out = tf.keras.layers.Dense(feature_number, activation='linear')(inputs)
    return out


def Inceptionv1_Module(inputs, filterB1_1, filterB2_1, filterB2_2, filterB3_1, filterB3_2, filterB4_1, i):
    # Inception Block i
    branch1x1 = Conv_2D_Block(inputs, filterB1_1, (1, 1), padding='valid')

    branch3x3 = Conv_2D_Block(inputs, filterB2_1, (1, 1), padding='valid')
    branch3x3 = Conv_2D_Block(branch3x3, filterB2_2, (3, 3))

    branch5x5 = Conv_2D_Block(inputs, filterB3_1, (1, 1), padding='valid')
    branch5x5 = Conv_2D_Block(branch5x5, filterB3_2, (5, 5))

    branch_pool = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(inputs)
    branch_pool = Conv_2D_Block(branch_pool, filterB4_1, (1, 1))
    out = tf.keras.layers.concatenate([branch1x1, branch3x3, branch5x5, branch_pool], axis=-1, name='Inception_Block_'+str(i))

    return out


def Inceptionv2_Module(inputs, filterB1_1, filterB2_1, filterB2_2, filterB3_1, filterB3_2, filterB3_3, filterB4_1, i):
    # Inception Block i
    branch1x1 = Conv_2D_Block(inputs, filterB1_1, (1, 1))

    branch3x3 = Conv_2D_Block(inputs, filterB2_1, (1, 1))
    branch3x3 = Conv_2D_Block(branch3x3, filterB2_2, (3, 3))

    branch3x3dbl = Conv_2D_Block(inputs, filterB3_1, (1, 1))
    branch3x3dbl = Conv_2D_Block(branch3x3dbl, filterB3_2, (3, 3))
    branch3x3dbl = Conv_2D_Block(branch3x3dbl, filterB3_3, (3, 3))

    branch_pool = tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(inputs)
    branch_pool = Conv_2D_Block(branch_pool, filterB4_1, (1, 1))

    out = tf.keras.layers.concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool], axis=-1, name='Inception_Block_'+str(i))

    return out


def Inception_Module_A(inputs, filterB1_1, filterB2_1, filterB2_2, filterB3_1, filterB3_2, filterB3_3, filterB4_1, i):
    # Inception Block i
    branch1x1 = Conv_2D_Block(inputs, filterB1_1, (1, 1))

    branch5x5 = Conv_2D_Block(inputs, filterB2_1, (1, 1))
    branch5x5 = Conv_2D_Block(branch5x5, filterB2_2, (5, 5))

    branch3x3dbl = Conv_2D_Block(inputs, filterB3_1, (1, 1))
    branch3x3dbl = Conv_2D_Block(branch3x3dbl, filterB3_2, (3, 3))
    branch3x3dbl = Conv_2D_Block(branch3x3dbl, filterB3_3, (3, 3))

    branch_pool = tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(inputs)
    branch_pool = Conv_2D_Block(branch_pool, filterB4_1, (1, 1))

    out = tf.keras.layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=-1, name='Inception_Block_A'+str(i))

    return out


def Inception_Module_B(inputs, filterB1_1, filterB2_1, filterB2_2, filterB3_1, filterB3_2, filterB3_3, filterB4_1, i):
    # Inception Block i
    branch1x1 = Conv_2D_Block(inputs, filterB1_1, (1, 1))

    branch7x7 = Conv_2D_Block(inputs, filterB2_1, (1, 1))
    branch7x7 = Conv_2D_Block(branch7x7, filterB2_2, (1, 7))
    branch7x7 = Conv_2D_Block(branch7x7, filterB2_2, (7, 1))

    branch7x7dbl = Conv_2D_Block(inputs, filterB3_1, 1)
    branch7x7dbl = Conv_2D_Block(branch7x7dbl, filterB3_2, (1, 7))
    branch7x7dbl = Conv_2D_Block(branch7x7dbl, filterB3_2, (7, 1))
    branch7x7dbl = Conv_2D_Block(branch7x7dbl, filterB3_3, (1, 7))
    branch7x7dbl = Conv_2D_Block(branch7x7dbl, filterB3_3, (7, 1))

    branch_pool = tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(inputs)
    branch_pool = Conv_2D_Block(branch_pool, filterB4_1, (1, 1))

    out = tf.keras.layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=-1, name='Inception_Block_B'+str(i))

    return out


def Inception_Module_C(inputs, filterB1_1, filterB2_1, filterB2_2, filterB3_1, filterB3_2, filterB3_3, filterB4_1, i):
    # Inception Block i
    branch1x1 = Conv_2D_Block(inputs, filterB1_1, (1, 1))

    branch3x3 = Conv_2D_Block(inputs, filterB2_1, (1, 1))
    branch3x3_2 = Conv_2D_Block(branch3x3, filterB2_2, (1, 3))
    branch3x3_3 = Conv_2D_Block(branch3x3, filterB2_2, (3, 1))

    branch3x3dbl = Conv_2D_Block(inputs, filterB3_1, (1, 1))
    branch3x3dbl = Conv_2D_Block(branch3x3dbl, filterB3_2, (1, 3))
    branch3x3dbl = Conv_2D_Block(branch3x3dbl, filterB3_2, (3, 1))
    branch3x3dbl_2 = Conv_2D_Block(branch3x3dbl, filterB3_3, (1, 3))
    branch3x3dbl_3 = Conv_2D_Block(branch3x3dbl, filterB3_3, (3, 1))

    branch_pool = tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(inputs)
    branch_pool = Conv_2D_Block(branch_pool, filterB4_1, (1, 1))

    out = tf.keras.layers.concatenate([branch1x1, branch3x3_2, branch3x3_3, branch3x3dbl_2, branch3x3dbl_3, branch_pool], axis=-1, name='Inception_Block_C'+str(i))

    return out


def Reduction_Block_A(inputs, filterB1_1, filterB1_2, filterB2_1, filterB2_2, filterB2_3, i):
    # Reduction Block A (i)
    branch3x3 = Conv_2D_Block(inputs, filterB1_1, (1, 1))
    branch3x3 = Conv_2D_Block(branch3x3, filterB1_2, (3, 3), strides=(2, 2))

    branch3x3dbl = Conv_2D_Block(inputs, filterB2_1, (1, 1))
    branch3x3dbl = Conv_2D_Block(branch3x3dbl, filterB2_2, (3, 3))
    branch3x3dbl = Conv_2D_Block(branch3x3dbl, filterB2_3, (3, 3), strides=(2, 2))

    branch_pool = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(inputs)
    out = tf.keras.layers.concatenate([branch3x3, branch3x3dbl, branch_pool], axis=-1, name='Reduction_Block_'+str(i))

    return out


def Reduction_Block_B(inputs, filterB1_1, filterB1_2, filterB2_1, filterB2_2, filterB2_3, i):
    # Reduction Block B (i)
    branch3x3 = Conv_2D_Block(inputs, filterB1_1, (1, 1))
    branch3x3 = Conv_2D_Block(branch3x3, filterB1_2, (3, 3), strides=(2, 2))

    branch3x3dbl = Conv_2D_Block(inputs, filterB2_1, (1, 1))
    branch3x3dbl = Conv_2D_Block(branch3x3dbl, filterB2_2, (1, 7))
    branch3x3dbl = Conv_2D_Block(branch3x3dbl, filterB2_2, (7, 1))
    branch3x3dbl = Conv_2D_Block(branch3x3dbl, filterB2_3, (3, 3), strides=(2, 2))

    branch_pool = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(inputs)
    out = tf.keras.layers.concatenate([branch3x3, branch3x3dbl, branch_pool], axis=-1, name='Reduction_Block_'+str(i))

    return out



# Inception V2, V3 and V4
class Inception:
    def __init__(self, input_shape, num_filters, problem_type='Regression',
                 output_nums=1, pooling='avg', dropout_rate=False, auxilliary_outputs=False):
        # length: Input Signal Length
        # model_depth: Depth of the Model
        # model_width: Width of the Model
        # kernel_size: Kernel or Filter Size of the Input Convolutional Layer
        # num_channel: Number of Channels of the Input Predictor Signals
        # problem_type: Regression or Classification
        # output_nums: Number of Output Classes in Classification mode and output features in Regression mode
        # pooling: Choose either 'max' for MaxPooling or 'avg' for Averagepooling
        # dropout_rate: If turned on, some layers will be dropped out randomly based on the selected proportion
        # auxilliary_outputs: Two extra Auxullary outputs for the Inception models, acting like Deep Supervision
        self.input_shape = input_shape
        self.num_filters = num_filters
        self.problem_type = problem_type
        self.output_nums = output_nums
        self.pooling = pooling
        self.dropout_rate = dropout_rate
        self.auxilliary_outputs = auxilliary_outputs

    def MLP(self, x):
        if self.pooling == 'avg':
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
        elif self.pooling == 'max':
            x = tf.keras.layers.GlobalMaxPooling2D()(x)
        # Final Dense Outputting Layer for the outputs
        x = tf.keras.layers.Flatten()(x)
        if self.dropout_rate:
            x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        outputs = tf.keras.layers.Dense(self.output_nums, activation='linear')(x)
        if self.problem_type == 'Classification':
            outputs = tf.keras.layers.Dense(self.output_nums, activation='softmax')(x)

        return outputs

    def Inception_v2(self):
        inputs = tf.keras.Input(self.input_shape)  # The input tensor
        # Stem: 56 x 64
        x = tf.keras.layers.SeparableConv2D(self.num_filters, kernel_size=7, strides=(2, 2), depth_multiplier=1, padding='same')(inputs)
        x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
        x = Conv_2D_Block(x, self.num_filters * 2, 1, padding='valid')
        x = Conv_2D_Block(x, self.num_filters * 6, 3, padding='valid')
        x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

        x = Inceptionv2_Module(x, 64, 64, 64, 64, 96, 96, 32, 1)  # Inception Block 1: 28 x 192
        x = Inceptionv2_Module(x, 64, 64, 96, 64, 96, 96, 64, 2)  # Inception Block 2: 28 x 256

        aux_output_0 = []
        if self.auxilliary_outputs:
            # Auxilliary Output 0
            aux_pool = tf.keras.layers.AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='valid')(x)
            aux_conv = Conv_2D_Block(aux_pool, 64, 1)
            aux_output_0 = self.MLP(aux_conv)

        x = Reduction_Block_A(x, 128, 160, 64, 96, 96, 1)  # Reduction Block 1: 28 x 320
        x = Inceptionv2_Module(x, 224, 64, 96, 96, 128, 128, 128, 3)  # Inception Block 3: 14 x 576
        x = Inceptionv2_Module(x, 192, 96, 128, 96, 128, 128, 128, 4)  # Inception Block 4: 14 x 576
        x = Inceptionv2_Module(x, 160, 128, 160, 128, 160, 160, 96, 5)  # Inception Block 5: 14 x 576
        x = Inceptionv2_Module(x, 96, 128, 192, 160, 192, 192, 96, 6)  # Inception Block 6: 14 x 576

        aux_output_1 = []
        if self.auxilliary_outputs:
            # Auxilliary Output 1
            aux_pool = tf.keras.layers.AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='valid')(x)
            aux_conv = Conv_2D_Block(aux_pool, 192, 1)
            aux_output_1 = self.MLP(aux_conv)

        x = Reduction_Block_A(x, 128, 192, 192, 256, 256, 2)  # Reduction Block 2: 14 x 576
        x = Inceptionv2_Module(x, 352, 192, 320, 160, 224, 224, 128, 7)  # Inception Block 7: 7 x 1024
        x = Inceptionv2_Module(x, 352, 192, 320, 192, 224, 224, 128, 8)  # Inception Block 8: 7 x 1024

        # Final Dense MLP Layer for the outputs
        final_output = self.MLP(x)
        # Create model.
        model = tf.keras.Model(inputs, final_output, name='Inception_v3')
        if self.auxilliary_outputs:
            model = tf.keras.Model(inputs, outputs=[final_output, aux_output_0, aux_output_1], name='Inception_v2')

        return model

    def Inception_v3(self):
        inputs = tf.keras.Input(self.input_shape)  # The input tensor
        # Stem
        x = Conv_2D_Block(inputs, self.num_filters, 3, strides=2, padding='valid')
        x = Conv_2D_Block(x, self.num_filters, 3, padding='valid')
        x = Conv_2D_Block(x, self.num_filters * 2, 3)
        x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

        x = Conv_2D_Block(x, self.num_filters * 2.5, 1, padding='valid')
        x = Conv_2D_Block(x, self.num_filters * 6, 3, padding='valid')
        x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

        # 3x Inception-A Blocks
        x = Inception_Module_A(x, 64, 48, 64, 64, 96, 96, 32, 1)  # Inception-A Block 1: 35 x 256
        x = Inception_Module_A(x, 64, 48, 64, 64, 96, 96, 64, 2)  # Inception-A Block 2: 35 x 256
        x = Inception_Module_A(x, 64, 48, 64, 64, 96, 96, 64, 3)  # Inception-A Block 3: 35 x 256

        aux_output_0 = []
        if self.auxilliary_outputs:
            # Auxilliary Output 0
            aux_pool = tf.keras.layers.AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='valid')(x)
            aux_conv = Conv_2D_Block(aux_pool, 64, 1)
            aux_output_0 = self.MLP(aux_conv)

        x = Reduction_Block_A(x, 64, 384, 64, 96, 96, 1)  # Reduction Block 1: 17 x 768

        # 4x Inception-B Blocks
        x = Inception_Module_B(x, 192, 128, 192, 128, 128, 192, 192, 1)  # Inception-B Block 1: 17 x 768
        x = Inception_Module_B(x, 192, 160, 192, 160, 160, 192, 192, 2)  # Inception-B Block 2: 17 x 768
        x = Inception_Module_B(x, 192, 160, 192, 160, 160, 192, 192, 3)  # Inception-B Block 3: 17 x 768
        x = Inception_Module_B(x, 192, 192, 192, 192, 192, 192, 192, 4)  # Inception-B Block 4: 17 x 768

        aux_output_1 = []
        if self.auxilliary_outputs:
            # Auxilliary Output 1
            aux_pool = tf.keras.layers.AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='valid')(x)
            aux_conv = Conv_2D_Block(aux_pool, 192, 1)
            aux_output_1 = self.MLP(aux_conv)

        x = Reduction_Block_B(x, 192, 320, 192, 192, 192, 2)  # Reduction Block 2: 8 x 1280

        # 2x Inception-C Blocks: 8 x 2048
        x = Inception_Module_C(x, 320, 384, 384, 448, 384, 384, 192, 1)  # Inception-C Block 1: 8 x 2048
        x = Inception_Module_C(x, 320, 384, 384, 448, 384, 384, 192, 2)  # Inception-C Block 2: 8 x 2048

        # Final Dense MLP Layer for the outputs
        final_output = self.MLP(x)
        # Create model.
        model = tf.keras.Model(inputs, final_output, name='Inception_v3')
        if self.auxilliary_outputs:
            model = tf.keras.Model(inputs, outputs=[final_output, aux_output_0, aux_output_1], name='Inception_v3')

        return model

    def Inception_v4(self):
        inputs = tf.keras.Input(self.input_shape)  # The input tensor
        # Stem
        x = Conv_2D_Block(inputs, 32, 3, strides=2, padding='valid')
        x = Conv_2D_Block(x, 32, 3, padding='valid')
        x = Conv_2D_Block(x, 64, 3)

        branch1 = Conv_2D_Block(x, 96, 3, strides=2, padding='valid')
        branch2 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
        x = tf.keras.layers.concatenate([branch1, branch2], axis=-1)

        branch1 = Conv_2D_Block(x, 64, 1)
        branch1 = Conv_2D_Block(branch1, 96, 3, padding='valid')
        branch2 = Conv_2D_Block(x, 64, 1)
        branch2 = Conv_2D_Block(branch2, 64, 7)
        branch2 = Conv_2D_Block(branch2, 96, 3, padding='valid')
        x = tf.keras.layers.concatenate([branch1, branch2], axis=-1)

        branch1 = Conv_2D_Block(x, 192, 3, padding='valid')
        branch2 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
        x = tf.keras.layers.concatenate([branch1, branch2], axis=1)

        # 4x Inception-A Blocks - 35 x 256
        for i in range(4):
            x = Inception_Module_A(x, 96, 64, 96, 64, 96, 96, 96, i)

        aux_output_0 = []
        if self.auxilliary_outputs:
            # Auxilliary Output 0
            aux_pool = tf.keras.layers.AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='valid')(x)
            aux_conv = Conv_2D_Block(aux_pool, 96, 1)
            aux_output_0 = self.MLP(aux_conv)

        x = Reduction_Block_A(x, 64, 384, 192, 224, 256, 1)  # Reduction Block 1: 17 x 768

        # 7x Inception-B Blocks - 17 x 768
        for i in range(7):
            x = Inception_Module_B(x, 384, 192, 256, 192, 224, 256, 128, i)

        aux_output_1 = []
        if self.auxilliary_outputs:
            # Auxilliary Output 1
            aux_pool = tf.keras.layers.AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='valid')(x)
            aux_conv = Conv_2D_Block(aux_pool, 128, 1)
            aux_output_1 = self.MLP(aux_conv)

        x = Reduction_Block_B(x, 192, 192, 256, 320, 320, 2)  # Reduction Block 2: 8 x 1280

        # 3x Inception-C Blocks: 8 x 2048
        for i in range(3):
            x = Inception_Module_C(x, 256, 384, 512, 384, 512, 512, 256, i)

        # Final Dense MLP Layer for the outputs
        final_output = self.MLP(x)
        # Create model.
        model = tf.keras.Model(inputs, final_output, name='Inception_v4')
        if self.auxilliary_outputs:
            model = tf.keras.layers.Model(inputs, outputs=[final_output, aux_output_0, aux_output_1], name='Inception_v4')

        return model