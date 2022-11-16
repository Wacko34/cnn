import numpy as np
import layers as lrs
import params as prm
from keras.datasets import mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()


filt_1 = np.random.randn(5, 1, 5, 5) * np.sqrt(1. / 5) # filling the filter with random numbers
bias_1 = np.random.randn(5, 1)
filt_2 = np.random.randn(20, 5, 5, 5) * np.sqrt(1. / 5) # filling the filter with random numbers
bias_2 = np.random.randn(20, 1)

INPUT_DIM = 2000
OUTPUT_DIM = 10
H_DIM = 4200
ALPHA = 0.0000005
BETA1 = 0.95
BETA2 = 0.99
correct = 0
N = 60000
NUM_EPOCHS = 1


W1 = np.random.rand(INPUT_DIM, H_DIM)
b1 = np.random.rand(1, H_DIM)

W2 = np.random.rand(H_DIM, OUTPUT_DIM)
b2 = np.random.rand(1, OUTPUT_DIM)

# adam params
v1 = np.zeros(filt_1.shape)
v2 = np.zeros(filt_2.shape)
v3 = np.zeros(W1.shape)
v4 = np.zeros(W2.shape)
bv1 = np.zeros(bias_1.shape)
bv2 = np.zeros(bias_2.shape)
bv3 = np.zeros(b1.shape)
bv4 = np.zeros(b2.shape)

s1 = np.zeros(filt_1.shape)
s2 = np.zeros(filt_2.shape)
s3 = np.zeros(W1.shape)
s4 = np.zeros(W2.shape)
bs1 = np.zeros(bias_1.shape)
bs2 = np.zeros(bias_2.shape)
bs3 = np.zeros(b1.shape)
bs4 = np.zeros(b2.shape)

W1 = (W1 - 0.5) * 2 * np.sqrt(1/INPUT_DIM)
b1 = (b1 - 0.5) * 2 * np.sqrt(1/INPUT_DIM)
W2 = (W2 - 0.5) * 2 * np.sqrt(1/H_DIM)
b2 = (b2 - 0.5) * 2 * np.sqrt(1/H_DIM)

# filt_1, filt_2, bias_1, bias_2, W1, W2, b1, b2 = prm.load_params(H_DIM, 0, 59999) 

# for sample in range(10000):
# 	if sample != 0 and sample % 1000 == 0 or sample == 59999:
# 			print("    i = ", sample)
# 	init_image = X_test[sample]
# 	init_image = np.expand_dims(init_image, axis=0)

# 	conv_image_1 = lrs.convolution(init_image, filt_1, bias_1, 1)
# 	relu_conv_image_1 = lrs.relu(conv_image_1)
# 	conv_image_2 = lrs.convolution(relu_conv_image_1, filt_2, bias_2, 1)
# 	relu_conv_image_2 = lrs.relu(conv_image_2)
# 	pooled_img = lrs.maxpool(relu_conv_image_2, 2, 2)
# 	(num_of_res_img, res_img_dim, _) = pooled_img.shape
# 	fc_vector_str = pooled_img.reshape(1, num_of_res_img * res_img_dim * res_img_dim) # flatten pooled layer
# 	(fc_dim, fc_lenght) = fc_vector_str.shape
# 	z = lrs.predict(fc_vector_str, Y_test[sample], W1, b1, W2, b2)
# 	y_pred = np.argmax(z)
# 	if y_pred == Y_test[sample]:
# 		correct += 1

# accuracy = correct / 10000 
# print("    Accuracy:", accuracy)


# pick a sample to plot
print("ALPHA = ", ALPHA)
print("H_DIM = ", H_DIM)
for ep in range(NUM_EPOCHS):
	print("epoch - ", ep)
	for i in range(N):

		#############################################################
		##################  Forward Backpropagation #################
		#############################################################

		correct = 0
		accuracy = 0
		# print("    i = ", i)
		sample = i
		init_image = X_train[sample]
		init_image = np.expand_dims(init_image, axis=0)
		# print("init image shape", init_image.shape) # (1, 28, 28)

		'''
		init image shape (1, 28, 28)
		conv1 image shape:  (5, 24, 24)
		relu conv image 1:  (5, 24, 24)
		conv2 image shape:  (20, 20, 20)
		pooled2 image shape:  (20, 10, 10)
		'''
		conv_image_1 = lrs.convolution(init_image, filt_1, bias_1, 1)

		relu_conv_image_1 = lrs.relu(conv_image_1)

		conv_image_2 = lrs.convolution(relu_conv_image_1, filt_2, bias_2, 1)

		relu_conv_image_2 = lrs.relu(conv_image_2)

		pooled_img = lrs.maxpool(relu_conv_image_2, 2, 2)

		#reshape 3d in 2d
		#flatten layer:
		(num_of_res_img, res_img_dim, _) = pooled_img.shape
		fc_vector_str = pooled_img.reshape(1, num_of_res_img * res_img_dim * res_img_dim) # flatten pooled layer

		(fc_dim, fc_lenght) = fc_vector_str.shape
		
		dE_dW1, dE_db1, dE_dW2, dE_db2, dE_dx_fc = lrs.dense(ALPHA, W1, b1, W2, b2, fc_lenght, OUTPUT_DIM, H_DIM, fc_vector_str, Y_train[sample])

		#############################################################
		#################  Backward Backpropagation #################
		#############################################################

		num_of_img_dim, img_dim = num_of_res_img, res_img_dim

		dE_dx_fc = dE_dx_fc.reshape(num_of_img_dim, img_dim, img_dim)

		de_dx_pool_2 =  lrs.maxpool_back(dE_dx_fc, conv_image_2, 2, 2)

		de_dx_pool_2 = lrs.relu_deriv(de_dx_pool_2)

		dE_dx_conv_2, dE_df_2, dE_db_2 = conv_back.conv_backward(de_dx_pool_2, relu_conv_image_1, filt_2, 1)

		dE_dx_conv_2 = lrs.relu_deriv(dE_dx_conv_2)

		dE_dx_conv_1, dE_df_1, dE_db_1 = conv_back.conv_backward(dE_dx_conv_2, init_image, filt_1, 1)

		#####################################################
		###################  SGD Update #####################
		#####################################################

		# filt_1 = filt_1 - ALPHA * dE_df_1
		# bias_1 = bias_1 - ALPHA * dE_db_1
		# filt_2 = filt_2 - ALPHA * dE_df_2
		# bias_2 = bias_2 - ALPHA * dE_db_2		
		# W1 = W1 - ALPHA * dE_dW1
		# b1 = b1 - ALPHA * dE_db1
		# W2 = W2 - ALPHA * dE_dW2
		# b2 = b2 - ALPHA * dE_db2

		#####################################################
		######################  Adam  #######################
		#####################################################
		

		v1 = BETA1 * v1 + (1 - BETA1) * dE_df_1 # momentum update
		s1 = BETA2 * s1 + (1 - BETA2) * dE_df_1**2 # RMSProp update
		filt_1 = filt_1 - ALPHA * v1 / np.sqrt(s1 + 1e-7) # combine momentum and RMSProp to perform update with Adam

		bv1 = BETA1 * bv1 + (1 - BETA1) * dE_db_1
		bs1 = BETA2 * bs1 + (1 - BETA2) * dE_db_1**2
		bias_1 = bias_1 - ALPHA * bv1 / np.sqrt(bs1 + 1e-7)

		v2 = BETA1 * v2 + (1 - BETA1) * dE_df_2
		s2 = BETA2 * s2 + (1 - BETA2) * dE_df_2**2
		filt_2 = filt_2 - ALPHA * v2 / np.sqrt(s2 + 1e-7)

		bv2 = BETA1 * bv2 + (1 - BETA1) * dE_db_2
		bs2 = BETA2 * bs2 + (1 - BETA2) * dE_db_2**2
		bias_2 = bias_2 - ALPHA * bv2 / np.sqrt(bs2 + 1e-7)

		v3 = BETA1 * v3 + (1 - BETA1) * dE_dW1
		s3 = BETA2 * s3 + (1 - BETA2) * dE_dW1**2
		W1 = W1 - ALPHA * v3 / np.sqrt(s3 + 1e-7)
		
		bv3 = BETA1 * bv3 + (1 - BETA1) * dE_db1
		bs3 = BETA2 * bs3 + (1 - BETA2) * dE_db1**2
		b1 = b1 - ALPHA * bv3 / np.sqrt(bs3 + 1e-7)
		
		v4 = BETA1 * v4 + (1 - BETA1) * dE_dW2
		s4 = BETA2 * s4 + (1 - BETA2) * dE_dW2**2
		W2 = W2 - ALPHA * v4 / np.sqrt(s4 + 1e-7)

		bv4 = BETA1 * bv4 + (1 - BETA1)*dE_db2
		bs4 = BETA2 * bs4 + (1 - BETA2) * dE_db2**2
		b2 = b2 - ALPHA * bv4 / np.sqrt(bs4 + 1e-7)

		# accuracy check
		if i != 0 and i % 1000 == 0 or i == 59999:
			print("    i = ", i)	
			# saving parameters every 1k elements
			prm.save_params(i, ep, H_DIM, filt_1, bias_1, filt_2, bias_2, W1, b1, W2, b2)
			for sample in range(1000):
				init_image = X_test[sample]
				init_image = np.expand_dims(init_image, axis=0)

				conv_image_1 = lrs.convolution(init_image, filt_1, bias_1, 1)
				relu_conv_image_1 = lrs.relu(conv_image_1)
				conv_image_2 = lrs.convolution(relu_conv_image_1, filt_2, bias_2, 1)
				relu_conv_image_2 = lrs.relu(conv_image_2)
				pooled_img = lrs.maxpool(relu_conv_image_2, 2, 2)
				(num_of_res_img, res_img_dim, _) = pooled_img.shape
				fc_vector_str = pooled_img.reshape(1, num_of_res_img * res_img_dim * res_img_dim) # flatten pooled layer
				(fc_dim, fc_lenght) = fc_vector_str.shape
				z = lrs.predict(fc_vector_str, Y_test[sample], W1, b1, W2, b2)
				y_pred = np.argmax(z)
				if y_pred == Y_test[sample]:
					correct += 1

			accuracy = correct / 1000 
			print("    Accuracy:", accuracy)

print("THE END.")