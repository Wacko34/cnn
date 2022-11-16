import numpy as np
import layers as lrs
import params as prm
from keras.datasets import mnist
import matplotlib.pyplot as plt

def predict(n, filt_1, bias_1, filt_2, bias_2, W1, b1, W2, b2):
	init_image = X_test[n]
	init_image = np.expand_dims(init_image, axis=0)
	conv_image_1 = lrs.convolution(init_image, filt_1, bias_1, 1)
	relu_conv_image_1 = lrs.relu(conv_image_1)
	conv_image_2 = lrs.convolution(relu_conv_image_1, filt_2, bias_2, 1)
	relu_conv_image_2 = lrs.relu(conv_image_2)
	pooled_img = lrs.maxpool(relu_conv_image_2, 2, 2)
	(num_of_res_img, res_img_dim, _) = pooled_img.shape
	fc_vector_str = pooled_img.reshape(1, num_of_res_img * res_img_dim * res_img_dim) # flatten pooled layer
	(fc_dim, fc_lenght) = fc_vector_str.shape
	z = lrs.predict(fc_vector_str, Y_test[n], W1, b1, W2, b2)
	y_pred = np.argmax(z)

	if y_pred == Y_test[n]:
		return y_pred
	else:
		return y_pred

# load params
(_, _), (X_test, Y_test) = mnist.load_data()
filt_1, filt_2, bias_1, bias_2, W1, W2, b1, b2 = prm.load_params(4200, 0, 59999) 
# accuracy = 92%

predict(9, filt_1, bias_1, filt_2, bias_2, W1, b1, W2, b2)


num_row = 4
num_col = 5
low = 0
high = 10000
n = np.random.randint(low, high)
fig = plt.figure(figsize=(30, 30))
t = 0
for i in range(num_row):
	for j in range(num_col):
		image = X_test[n + t]	
		answ = predict(n + t, filt_1, bias_1, filt_2, bias_2, W1, b1, W2, b2)

		fig.add_subplot(num_row, num_col, t+1)
		plt.imshow(image, cmap='gray')
		plt.axis('off')
		title = "predict = " + str(answ) + ", right = " + str(Y_test[n + t])
		plt.title(title)

		t = t + 1

plt.show()