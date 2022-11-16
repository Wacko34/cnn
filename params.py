import numpy as np

def save_params(i, ep, H_DIM, filt_1, bias_1, filt_2, bias_2, W1, b1, W2, b2):
	path_filt_1 = "params" + "/1/" + str(H_DIM) + "/filt_1/" + str(ep) + "_" + str(i)
	path_filt_2 = "params" + "/1/" + str(H_DIM) + "/filt_2/" + str(ep) + "_" + str(i)
	path_bias_1 = "params" + "/1/" + str(H_DIM) + "/bias_1/" + str(ep) + "_" + str(i)
	path_bias_2 = "params" + "/1/" + str(H_DIM) + "/bias_2/" + str(ep) + "_" + str(i)
	path_W1 = "params" + "/1/" + str(H_DIM) + "/W1/" + str(ep) + "_" + str(i)
	path_W2 = "params" + "/1/" + str(H_DIM) + "/W2/" + str(ep) + "_" + str(i)
	path_b1 = "params" + "/1/" + str(H_DIM) + "/b1/" + str(ep) + "_" + str(i)
	path_b2 = "params" + "/1/" + str(H_DIM) + "/b2/" + str(ep) + "_" + str(i)
	np.save(path_filt_1, filt_1)
	np.save(path_filt_2, filt_2)
	np.save(path_bias_1, bias_1)
	np.save(path_bias_2, bias_2)
	np.save(path_W1, W1)
	np.save(path_W2, W2)
	np.save(path_b1, b1)
	np.save(path_b2, b2)

def load_params(H_DIM, ep, i):
	path_filt_1 = "params/filt_1.npy"
	path_filt_2 = "params/filt_2.npy"
	path_bias_1 = "params/bias_1.npy"
	path_bias_2 = "params/bias_2.npy"
	path_W1 = "params/W1.npy"
	path_W2 = "params/W2.npy"
	path_b1 = "params/b1.npy"
	path_b2 = "params/b2.npy"
	filt_1 = np.load(path_filt_1)
	filt_2 = np.load(path_filt_2)
	bias_1 = np.load(path_bias_1)
	bias_2 = np.load(path_bias_2)
	W1 = np.load(path_W1)
	W2 = np.load(path_W2)
	b1 = np.load(path_b1)
	b2 = np.load(path_b2)
	return filt_1, filt_2, bias_1, bias_2, W1, W2, b1, b2