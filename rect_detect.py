import os
import numpy as np
import tensorflow as tf
from tensorflow import ConfigProto,GPUOptions,Session
from keras.backend.tensorflow_backend import set_session
from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D,concatenate,AtrousConvolution2D
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from data import *
import keras.backend.tensorflow_backend as KTF
from tensorflow.contrib.keras import layers

class myUnet(object):
	def __init__(self):
		self.data_path='/data4/liyh/Unet-master11/raw/'
		self.img_type='tif'
		self.test_path='/data4/liyh/Unet-master11/test/'
		self.validation_path='/data4/liyh/Unet-master11/validate/'

	def get_unet(self):
		inputs = Input([None, None,1])
		conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
		print("conv1 shape:", conv1.shape)
		conv1 = AtrousConvolution2D(64, 3,atrous_rate=(2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)

		#conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
		print("conv1 shape:", conv1.shape)
		pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv1)
		print("pool1 shape:", pool1.shape)

		conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
		print ("conv2 shape:",conv2.shape)
		conv2 = AtrousConvolution2D(128, 3,atrous_rate=(2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)

		#conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
		print ("conv2 shape:",conv2.shape)
		pool2 = MaxPooling2D(pool_size=(2, 2),strides=(2, 2),padding='valid')(conv2)
		print ("pool2 shape:",pool2.shape)

		conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
		print ("conv3 shape:",conv3.shape)
		conv3 = AtrousConvolution2D(256, 3,atrous_rate=(2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
		#conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
		print ("conv3 shape:",conv3.shape)
		pool3 = MaxPooling2D(pool_size=(2, 2),strides=(2, 2),padding='valid')(conv3)
		print ("pool3 shape:",pool3.shape)

		conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
		conv4 = AtrousConvolution2D(512, 3,atrous_rate=(2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
		drop4 = Dropout(0.5)(conv4)
		pool4 = MaxPooling2D(pool_size=(2, 2),strides=(2, 2),padding='valid')(drop4)

		conv5_1 = AtrousConvolution2D(1024, 3,atrous_rate=(1,1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
		conv5_2 = AtrousConvolution2D(1024, 3,atrous_rate=(2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5_1)
		#drop5 = Dropout(0.5)(conv5_3)

		up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv5_2))
		#merge6 = merge([drop4,up6], mode = 'concat', concat_axis = 3)

		merge6 = concatenate([drop4, up6],axis=3)
		conv6 = AtrousConvolution2D(512, 3,atrous_rate=(1,1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
		conv6 = AtrousConvolution2D(512, 3,atrous_rate=(2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

		up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
		# merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 3)

		#merge7 = concatenate([conv3, up7],axis=3)
		conv7 = AtrousConvolution2D(256, 3,atrous_rate=(1,1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up7)
		conv7 = AtrousConvolution2D(256, 3,atrous_rate=(2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

		up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
		# merge8 = merge([conv2,up8], mode = 'concat', concat_axis = 3)

		#merge8 = concatenate([conv2,up8],axis=3)
		conv8 = AtrousConvolution2D(128, 3,atrous_rate=(1,1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up8)
		conv8 = AtrousConvolution2D(128, 3,atrous_rate=(2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

		up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
		# merge9 = merge([conv1,up9], mode = 'concat', concat_axis = 3)

		#merge9 = concatenate([conv1,up9],axis=3)
		conv9 = AtrousConvolution2D(64, 3,atrous_rate=(1,1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up9)
		conv9 = AtrousConvolution2D(64, 3,atrous_rate=(2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
		conv9 = AtrousConvolution2D(2, 3,atrous_rate=(1,1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
		conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)


		model = Model(input = inputs, output = conv10)
		model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
		return model

	def train(self):
		print("loading data")
		print("loading data done")
		model = self.get_unet()
		print("got unet")
		model_checkpoint = ModelCheckpoint('my_unet.hdf5', monitor='loss',verbose=1, save_best_only=True)
		print('Fitting model...')
		model.fit_generator(data_generator(self.data_path),steps_per_epoch=369,epochs=10,validation_data=validate_generator(self.validation_path),validation_steps=100,callbacks=[model_checkpoint])
		model.load_weights('./my_unet.hdf5')
		tests = glob.glob(self.test_path + "/*." + self.img_type)
		for imgname in tests:
			midname = imgname[imgname.rindex("/") + 1:]
			print(midname)
			imgs_test = load_img(self.test_path + midname, color_mode = 'grayscale')
			imgs_test = img_to_array(imgs_test)
			imgs_test = imgs_test.astype('float32')
			imgs_test /= 255
			mean = imgs_test.mean(axis=0)
			imgs_test -= mean
			nx = np.shape(imgs_test)[1]
			ny = np.shape(imgs_test)[0]
			imgs_test = np.array(imgs_test)
			imgs_test=imgs_test.reshape(1,ny, nx, 1)
			imgs_mask_test = model.predict_on_batch(imgs_test)
			imgs_mask_test = np.array(imgs_mask_test)
			imgs_mask_test[imgs_mask_test >= 0.5] = 1
			imgs_mask_test[imgs_mask_test < 0.5] = 0
			imgs_mask_test = imgs_mask_test.reshape(ny, nx, 1)
			img = array_to_img(imgs_mask_test)
			img.save("/data4/liyh/Unet-master11/results/{}".format(midname))

	def save_img(self):
		print("array to image")
		imgs = np.load('./results/imgs_mask_test.npy')
		for i in range(imgs.shape[0]):
			img = imgs[i]
			img = array_to_img(img)
			img.save("./results/%d.png"%(i))


if __name__ == '__main__':
	os.environ['CUDA_VISIBLE_DEVICES']=str(5)
	config=tf.ConfigProto(allow_soft_placement=True)
	config.gpu_options.per_process_gpu_memory_fraction=0.9
	session=tf.Session(config=config)
	KTF.set_session(session)
	myunet = myUnet()
	myunet.train()
