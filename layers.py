import numpy as np

def convolution(image, filt, bias, s=1):
	'''
	Confolves `filt` over `image` using stride `s`
	'''

	(num_of_filt, num_of_filt_dim, filt_dim, _) = filt.shape # filter dimensions
	num_of_img_dim, img_dim, _ = image.shape # image dimensions
	
	out_dim = int((img_dim - filt_dim)/s)+1 # calculate output dimensions
	
	# ensure that the filter dimensions match the dimensions of the input image
	assert num_of_img_dim == num_of_filt_dim, "Dimensions of filter must match dimensions of input image"

	out = np.zeros((num_of_filt,out_dim,out_dim)) # create the matrix to hold the values of the convolution operation
	
	# convolve each filter over the image
	for curr_f in range(num_of_filt):
		curr_y = out_y = 0
		# move filter vertically across the image
		while curr_y + filt_dim <= img_dim:
			curr_x = out_x = 0
			# move filter horizontally across the image 
			while curr_x + filt_dim <= img_dim:
				# perform the convolution operation and add the bias
				# out[curr_f, out_y, out_x] = np.sum(filt[curr_f] * image[:, curr_y:curr_y + filt_dim, curr_x:curr_x + filt_dim]) + bias[curr_f]
				out[curr_f, out_y, out_x] = np.sum(filt[curr_f] * image[:, curr_y:curr_y + filt_dim, curr_x:curr_x + filt_dim]) + bias[curr_f]
				curr_x += s
				out_x += 1
			curr_y += s
			out_y += 1
	return out

def conv_backward(dE_dx, image, filt, stride):
	'''
	Backpropagation through a convolutional layer. 
	'''
	(n_f, n_c, f, _) = filt.shape
	(_, orig_dim, _) = image.shape
	## initialize derivatives
	dE_dX_out = np.zeros(image.shape) 
	dE_dF = np.zeros(filt.shape)
	dE_db = np.zeros((n_f,1))

	for curr_f in range(n_f):
		# loop through all filters
		curr_y = out_y = 0
		while curr_y + f <= orig_dim:
			curr_x = out_x = 0
			while curr_x + f <= orig_dim:
				dE_dF[curr_f] += dE_dx[curr_f, out_y, out_x] * image[:, curr_y:curr_y+f, curr_x:curr_x+f]
				dE_dX_out[:, curr_y:curr_y+f, curr_x:curr_x+f] += dE_dx[curr_f, out_y, out_x] * filt[curr_f] 
				curr_x += stride
				out_x += 1
			curr_y += stride
			out_y += 1
		dE_db[curr_f] = np.sum(dE_dx[curr_f])

	return dE_dX_out, dE_dF, dE_db

def relu(orig):
	num_of_img_dim, img_dim, _ = orig.shape

	for curr_dim in range(num_of_img_dim):
		for y in range(img_dim):
			for x in range(img_dim):
				orig[curr_dim][y][x] = np.maximum(orig[curr_dim][y][x], 0)
	return orig

def relu_deriv(orig):
	num_of_img_dim, img_dim, _ = orig.shape
	for curr_dim in range(num_of_img_dim):
		for y in range(img_dim):
			for x in range(img_dim):
				if (orig[curr_dim][y][x] >= 0):
					orig[curr_dim][y][x] = 1
				else:
					orig[curr_dim][y][x] = 0
	return orig

def maxpool(image, f=2, s=2):
	'''
	Downsample input `image` using a kernel size of `f` and a stride of `s`
	'''

	n_c, h_prev, w_prev = image.shape

	# calculate output dimensions after the maxpooling operation.
	h = int((h_prev - f)/s)+1 
	w = int((w_prev - f)/s)+1

	# create a matrix to hold the values of the maxpooling operation.
	downsampled = np.zeros((n_c, h, w)) 

	# slide the window over every part of the image using stride s. Take the maximum value at each step.
	for i in range(n_c):
		curr_y = out_y = 0
		# slide the max pooling window vertically across the image
		while curr_y + f <= h_prev:
			curr_x = out_x = 0
			# slide the max pooling window horizontally across the image
			while curr_x + f <= w_prev:
				downsampled[i, out_y, out_x] = np.max(image[i, curr_y:curr_y+f, curr_x:curr_x+f])
				curr_x += s
				out_x += 1
			curr_y += s
			out_y += 1
			
	return downsampled

def maxpool_back(dE_dx, orig, filt_size, stride):
	(num_of_dim, orig_dim, _) = orig.shape
	out = np.zeros(orig.shape)

	for curr_dim in range(num_of_dim):
		curr_y = out_y = 0
		while curr_y + filt_size <= orig_dim:
			curr_x = out_x = 0
			while curr_x + filt_size <= orig_dim:
				arr = orig[curr_dim, curr_y:curr_y + filt_size, curr_x:curr_x + filt_size]
				i, j= np.where(arr==np.nanmax(arr))
				max_y = i[0]
				max_x = j[0]
				out[curr_dim, curr_y + max_y, curr_x + max_x] = dE_dx[curr_dim, out_y, out_x]
				curr_x += stride
				out_x += 1
			curr_y += stride
			out_y += 1
	return out

def relu(t):
	return np.maximum(t, 0)

def softmax(t):
	t = t - np.max(t)
	out = np.exp(t)
	return out / np.sum(out)

def sparse_cross_entropy(z, y):
	return -np.log(z[0, y])

def to_full(y, num_classes):
	y_full = np.zeros((1, num_classes))
	y_full[0, y] = 1
	return y_full

def relu_deriv(t):
    return (t >= 0).astype(float)

def dense(ALPHA, W1, b1, W2, b2, INPUT_DIM, OUTPUT_DIM, H_DIM, x, y):
	# Forward
	t1 = x @ W1 + b1
	h1 = relu(t1)
	t2 = h1 @ W2 + b2
	z = softmax(t2)
	E = sparse_cross_entropy(z, y)

	#Backward
	y_full = to_full(y, OUTPUT_DIM)

	dE_dt2 = z - y_full
	dE_dW2 = h1.T @ dE_dt2
	dE_db2 = dE_dt2
	dE_dh1 = dE_dt2 @ W2.T
	dE_dt1 = dE_dh1 * relu_deriv(t1)
	dE_dW1 = x.T @ dE_dt1 
	dE_db1 = dE_dt1
	dE_dx = dE_dt1 @ W1.T

	return dE_dW1, dE_db1, dE_dW2, dE_db2, dE_dx

def predict(x, y, W1, b1, W2, b2):
	t1 = x @ W1 + b1
	h1 = relu(t1)
	t2 = h1 @ W2 + b2
	z = softmax(t2)
	return z