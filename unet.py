import numpy as np
from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D,Conv2DTranspose
from keras.layers import add, concatenate, BatchNormalization, Activation
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, TensorBoard
from keras import backend as keras
from data import dataProcess
from keras.preprocessing.image import array_to_img
from PIL import Image as pil_image
from keras.preprocessing.image import *
import tensorflow as tf
from keras.losses import *
from xlwt import *
import xlwt
import xlrd
from xlutils.copy import copy  
import SimpleITK as sitk


NAME='a'
class myUnet(object):

	def __init__(self, img_rows = 128, img_cols = 128):

		self.img_rows = img_rows
		self.img_cols = img_cols

	def load_data(self):

#		mydata = dataProcess(self.img_rows, self.img_cols,data_path='D:/Python/unet-master/data/D'
#			,label_path='D:/Python/unet-master/data/L',test_path='D:/Python/unet-master/data/T',
#			npy_path='D:/Python/unet-master/data')
		mydata = dataProcess(self.img_rows,self.img_cols,data_path='F:/Octave/code/GAN/Train',
			label_path='F:/Octave/code/GAN/Train_lab',
			test_path='F:/Octave/code/GAN/Data/'+NAME,
			test_label_path='F:/Octave/code/GAN/Data/'+ NAME +'_lab',
			npy_path='F:/Octave/code/GAN/NPY',img_type='bmp')
		mydata.create_train_data()
		mydata.create_test_data()

		
		imgs_train, imgs_mask_train = mydata.load_train_data()
		imgs_test,imgs_mask_test,name = mydata.load_test_data()

		return imgs_train, imgs_mask_train, imgs_test,imgs_mask_test,name
		
	def get_unet(self):

		inputs = Input((self.img_rows, self.img_cols,1))
		print('input size: ',inputs.shape)

		
		conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
		print ("conv1 shape:",conv1.shape)
		conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)  		
		print ("conv1 shape:",conv1.shape)
#		pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
		pool1 = Conv2D(64, 2, strides=(2,2), activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv1)
		print ("pool1 shape:",pool1.shape)
		print('\n')
		
		conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
		print ("conv2 shape:",conv2.shape)
		conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
		print ("conv2 shape:",conv2.shape)
#		pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
		pool2 = Conv2D(128, 2, strides=(2,2), activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv2)
		print ("pool2 shape:",pool2.shape)
		print('\n')

		conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
		print ("conv3 shape:",conv3.shape)
		conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
		print ("conv3 shape:",conv3.shape)
#		drop3 = Dropout(0.5)(conv3)
#		pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
		pool3 = Conv2D(256, 2, strides=(2,2), activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv3)
		print ("pool3 shape:",pool3.shape)
		print('\n')
		
		conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
		print ("conv4 shape:",conv4.shape)
		conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
		print ("conv4 shape:",conv4.shape)
#		drop4 = Dropout(0.5)(conv4)		
#		pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
		pool4 = Conv2D(512, 2, strides=(2,2), activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv4)
		print ("pool4 shape:",pool4.shape)
		print('\n')

		conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
		print ("conv5 shape:",conv5.shape)
		conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
		print ("conv5 shape:",conv5.shape)
		drop5 = Dropout(0.5)(conv5)
		print('\n')
		
#		up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
		up6 = Conv2DTranspose(512, 3, strides=(2, 2), padding='same', kernel_initializer = 'he_normal')(drop5)
		print ("up6 shape:",up6.shape)
		merge6 = concatenate([conv4,up6])	
		print ("merge6 shape:",merge6.shape)
		conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
		print ("conv6 shape:",conv6.shape)
		conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
		print ("conv6 shape:",conv6.shape)
		print('\n')
		
#		up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop4))
		up7 = Conv2DTranspose(256, 3, strides=(2, 2), padding='same', kernel_initializer = 'he_normal')(conv6)
		print ("up7 shape:",up7.shape)
		merge7 = concatenate([conv3,up7])
		print ("merge7 shape:",merge7.shape)
		conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
		print ("conv7 shape:",conv7.shape)
		conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
		print ("conv7 shape:",conv7.shape)
		print('\n')

#		up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
		up8 = Conv2DTranspose(128, 3, strides=(2, 2), padding='same', kernel_initializer = 'he_normal')(conv7)
		print ("up8 shape:",up8.shape)
		merge8 = concatenate([conv2,up8])
		print ("merge8 shape:",merge8.shape)
		conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8) 
		print ("conv8 shape:",conv8.shape)
		conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
		print ("conv8 shape:",conv8.shape)
		print('\n')

#		up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
		up9 = Conv2DTranspose(64, 3, strides=(2, 2), padding='same', kernel_initializer = 'he_normal')(conv8)
		print ("up9 shape:",up9.shape)
		merge9 = concatenate([conv1,up9])
		print ("merge9 shape:",merge9.shape)
		conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)	
		print ("conv9 shape:",conv9.shape)
		conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
		print ("conv9 shape:",conv9.shape)
		conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
		print ("conv9 shape:",conv9.shape)
		conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
		print ("conv10 shape:",conv10.shape)
		print('\n')
		
		
		model = Model(inputs = inputs, outputs = conv10)
#		print(type(model))
		model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
#		model.compile(optimizer = Adam(lr = 1e-4), loss = 'dice_coef_loss', metrics = ['dice_coef'])

		return model
		


	def train(self):
		
		print("loading data")
		imgs_train, imgs_mask_train, imgs_test, imgs_mask_test,Name= self.load_data()
		print("loading data done")
		model = self.get_unet()
		print("got unet")
		print(imgs_train.shape)
		print(imgs_mask_train.shape)
		print(imgs_test.shape)
		print(imgs_mask_test.shape)


		
#		tb = TensorBoard(log_dir='C:/Users/ppk/Code/GAN/logs', batch_size=1)
		earlyStopping = EarlyStopping(monitor='loss', min_delta=0.001, patience=2, verbose=1)
		model_checkpoint = ModelCheckpoint('F:/Octave/code/GAN/unet.hdf5', monitor='loss',verbose=1, save_best_only=True)
		print('Fitting model...')
		print(len(model.layers))
#		model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
		model.fit(imgs_train, imgs_mask_train, batch_size=10,epochs=10, verbose=1, shuffle=True, 
			callbacks=[model_checkpoint,earlyStopping])		
		
		

	def predict(self):	
		print('predict test data')
		model=load_model('F:/Octave/code/GAN/unet.hdf5')
		imgs_pred_test=np.zeros((imgs_test.shape[0],128,128,1))

		rb = xlrd.open_workbook('F:/Octave/code/GAN/result.xls', formatting_info=True)  
		wb = copy(rb)  
		ws = wb.get_sheet(0)  
#		workbook=xlwt.Workbook(encoding='utf-8')
#		booksheet=workbook.add_sheet('Sheet 1', cell_overwrite_ok=True)

		for i in range(imgs_test.shape[0]):
			Temp = imgs_test[i,:,:,:]
			Temp.resize(1,128,128,1)
			imgs_pred_test[i,:,:,:] = model.predict(Temp,batch_size=1, verbose=1)
			
		
		imgs_pred_test[imgs_pred_test > 0.5]=1
		imgs_pred_test[imgs_pred_test <= 0.5]=0
		np.save('imgs_mask_test.npy', imgs_pred_test)
		
		temp = np.zeros((imgs_test.shape[0],128,128))
		for i in range(imgs_test.shape[0]):
			Index = Name.index(i+1)
			temp[i,:,:] = imgs_pred_test[Index,:,:,0]

		temp2 = np.zeros((103,320,232),dtype=np.int16)
#		index_l,index_r=57,79
#		index_l,index_r=58,101
#		index_l,index_r=32,87
#		index_l,index_r=37,58
#		index_l,index_r=7,38
#		index_l,index_r=32,98
#		index_l,index_r=60,93
#		index_l,index_r=29,53
#		index_l,index_r=75,99

		index_l,index_r=37,77

		#Save the result
		temp2[index_l:index_r,99:99+128,49:49+128] = temp
		Image = sitk.GetImageFromArray(temp2)
		sitk.WriteImage(Image,"F:/Octave/code/GAN/Result/predict.mha")

				
		#compute Dice, PM, CR, ASSD
		Dice=0
		PM=0
		CR=0
		ASSD=[0 for i in range(imgs_test.shape[0])]
		print('Dice: ',Dice)

		sum_pred = 0
		sum_true = 0
		sum_intersection = 0
		n = 0
		col_pred = np.zeros((imgs_test.shape[0],128,128))
		for i in range(imgs_test.shape[0]):
			Index = Name.index(i+1)

			Pred = imgs_pred_test[Index,:,:,:]
			Mask = imgs_mask_test[Index,:,:,:]
    		
			y_pred_f = Pred.flatten()
			y_true_f = Mask.flatten()
			intersection = sum(y_true_f * y_pred_f)
			sum_intersection = sum_intersection + intersection
			sum_pred = sum_pred + sum(y_pred_f)
			sum_true = sum_true + sum(y_true_f)

			temp = y_pred_f - y_true_f
			for j in range(y_true_f.shape[0]):
				if temp[j]==1:
					n=n+1

			#save the images 
			temp=imgs_pred_test[Index,:,:,0]
			temp.resize(128,128,1)
			pic=array_to_img(temp,data_format='channels_last')
			pic.save('F:/Octave/code/GAN/Result/'+str(Name[Index])+'.jpg')

			#get mutillabel
			temp = y_pred_f - y_true_f
			for j in range(y_true_f.shape[0]):
				if (y_pred_f[j]==1 and y_true_f[j]==1):
					temp[j]=2
				elif temp[j]==1:
					temp[j]=4
				elif temp[j]==-1 :
					temp[j]=3
			temp.resize(128,128)
			col_pred[i,:,:]=temp	

		Dice = 2 * sum_intersection/(sum_true + sum_pred)
		PM = sum_intersection/sum_true
		CR = (sum_intersection - 0.5*n)/sum_true

		#Get the ASSD			
#		GT = sitk.ReadImage('F:/Octave/code/Gan/GT/'+NAME+'.mha')
#		Array = sitk.GetArrayFromImage(GT)
#		Image = sitk.GetImageFromArray(Array)
#		sitk.WriteImage(Image,"F:/Octave/code/GAN/Result/GT.mha")
#		Temp = os.popen('E:/C++/Code/Temp/ISLESevaluation-master/x64/Debug/ISLESevaluation \
#				F:/Octave/code/GAN/Result/GT.mha \
#				F:/Octave/code/GAN/Result/predict.mha')
#		ASSD = float(Temp.read()[0:-1])
		
		#save the result to an excel file
		x_index=1
		ws.write(x_index,0,NAME)

		ws.write(0,1,'Dice')
		ws.write(x_index,1,Dice)

		ws.write(0,2,'PM')
		ws.write(x_index,2,PM)

		ws.write(0,3,'CR')
		ws.write(x_index,3,CR)

		ws.write(0,4,'ASSD')
#		ws.write(x_index,4,ASSD)
		wb.save('F:/Octave/code/GAN/result.xls')  


		temp2[index_l:index_r,99:99+128,49:49+128] = col_pred
		Image = sitk.GetImageFromArray(temp2)
		sitk.WriteImage(Image,"F:/Octave/code/GAN/Result/predict_col.mha")			
		
		print('Dice: ',Dice)
		


if __name__ == '__main__':
	NAME = 'ZhouLiangYong'
#	config = tf.ConfigProto()
#	config.gpu_options.allow_growth = True
#	config.gpu_options.per_process_gpu_memory_fraction = 0.4

#with tf.Session(config=config) as sess:
	myunet = myUnet()
	myunet.train()
#	myunet.predict()








