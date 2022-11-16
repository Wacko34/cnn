import numpy as np
import layers as lrs
import params as prm
from keras.datasets import mnist
import matplotlib.pyplot as plt

# load params
(_, _), (X_test, Y_test) = mnist.load_data()
filt_1, filt_2, bias_1, bias_2, W1, W2, b1, b2 = prm.load_params(4200, 0, 59999) 
# accuracy = 92%

low = 0
high = 10000
n = np.random.randint(low, high)
print(Y_test[n])

init_image = X_test[n]
init_image = np.expand_dims(init_image, axis=0)
conv_image_1 = lrs.convolution(init_image, filt_1, bias_1, 1)
relu_conv_image_1 = lrs.relu(conv_image_1)
conv_image_2 = lrs.convolution(relu_conv_image_1, filt_2, bias_2, 1)
relu_conv_image_2 = lrs.relu(conv_image_2)
pooled_img = lrs.maxpool(relu_conv_image_2, 2, 2)
(num_of_res_img, res_img_dim, _) = pooled_img.shape
fc_vector_str = pooled_img.reshape(1, num_of_res_img * res_img_dim * res_img_dim)
(fc_dim, fc_lenght) = fc_vector_str.shape
z = lrs.predict(fc_vector_str, Y_test[n], W1, b1, W2, b2)
y_pred = np.argmax(z)

num_row = 5
num_col = 3
fig = plt.figure(figsize=(30, 30))

init_image = np.squeeze(init_image, axis=0)
fig.add_subplot(num_row, num_col, 7)
plt.imshow(init_image, cmap='gray')
plt.axis('off')
plt.title('Initial image')

f_0 = np.squeeze(filt_1[0, :, :, :], axis=0)
f_1 = np.squeeze(filt_1[1, :, :, :], axis=0)
f_2 = np.squeeze(filt_1[2, :, :, :], axis=0)
f_3 = np.squeeze(filt_1[3, :, :, :], axis=0)
f_4 = np.squeeze(filt_1[4, :, :, :], axis=0)

fig.add_subplot(num_row, num_col, 2)
plt.imshow(f_0, cmap='gray')
plt.axis('off')
plt.title('f_0')

fig.add_subplot(num_row, num_col, 5)
plt.imshow(f_1, cmap='gray')
plt.axis('off')
plt.title('f_1')

fig.add_subplot(num_row, num_col, 8)
plt.imshow(f_2, cmap='gray')
plt.axis('off')
plt.title('f_2')

fig.add_subplot(num_row, num_col, 11)
plt.imshow(f_3, cmap='gray')
plt.axis('off')
plt.title('f_3')

fig.add_subplot(num_row, num_col, 14)
plt.imshow(f_4, cmap='gray')
plt.axis('off')
plt.title('f_4')

conv_img_0 = conv_image_1[0, :, :]
conv_img_1 = conv_image_1[1, :, :]
conv_img_2 = conv_image_1[2, :, :]
conv_img_3 = conv_image_1[3, :, :]
conv_img_4 = conv_image_1[4, :, :]

fig.add_subplot(num_row, num_col, 3)
plt.imshow(conv_img_0, cmap='gray')
plt.axis('off')
plt.title('Convolved image 0')

fig.add_subplot(num_row, num_col, 6)
plt.imshow(conv_img_1, cmap='gray')
plt.axis('off')
plt.title('Convolved image 1')

fig.add_subplot(num_row, num_col, 9)
plt.imshow(conv_img_2, cmap='gray')
plt.axis('off')
plt.title('Convolved image 2')

fig.add_subplot(num_row, num_col, 12)
plt.imshow(conv_img_3, cmap='gray')
plt.axis('off')
plt.title('Convolved image 3')

fig.add_subplot(num_row, num_col, 15)
plt.imshow(conv_img_4, cmap='gray')
plt.axis('off')
plt.title('Convolved image 4')

plt.show()