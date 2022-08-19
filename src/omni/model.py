from keras.models import *
from keras.layers import *

def conv_block(tensor, n_filters, size=3, padding='same', initializer='he_normal'):
    x = Conv2D(filters=n_filters, kernel_size=(size, size), padding=padding, kernel_initializer=initializer)(tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=n_filters, kernel_size=(size, size), padding=padding, kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def deconv_block(tensor, residual, n_filters, size=3, padding='same', strides=(2, 2)):
    y = Conv2DTranspose(n_filters, kernel_size=(size, size), strides=strides, padding=padding)(tensor)
    y = concatenate([y, residual], axis=3)
    y = conv_block(y, n_filters)
    return y

def deconv_coarse(tensor, n_filters, size=3, padding='same', initializer='he_normal'):
    x = Conv2D(filters=n_filters, kernel_size=(size, size), padding=padding, kernel_initializer=initializer)(tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def unet(img_size, filter_size, n_classes, pretrained_weights=None):

    input_layer = Input(img_size + (3,))
    
    conv1 = conv_block(input_layer, n_filters=filter_size)
    conv1_out = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = conv_block(conv1_out, n_filters=filter_size*2)
    conv2_out = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = conv_block(conv2_out, n_filters=filter_size*4)
    conv3_out = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = conv_block(conv3_out, n_filters=filter_size*8)
    conv4_out = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv4_out = Dropout(0.5)(conv4_out)

    conv5 = conv_block(conv4_out, n_filters=filter_size*16)
    conv5 = Dropout(0.5)(conv5)

    deconv6 = deconv_block(conv5, residual=conv4, n_filters=filter_size*8)
    deconv6 = Dropout(0.5)(deconv6)
    deconv7 = deconv_block(deconv6, residual=conv3, n_filters=filter_size*4)
    deconv8 = deconv_block(deconv7, residual=conv2, n_filters=filter_size*2)
    deconv9 = deconv_block(deconv8, residual=conv1, n_filters=filter_size)

    output_layer = Conv2D(filters=n_classes, kernel_size=(1, 1))(deconv9)
    output_layer = BatchNormalization()(output_layer)
    output_layer = Activation('softmax')(output_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    return model
