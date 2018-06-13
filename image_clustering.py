
"""

# IMAGE CLUSTERING

# Below is the function that does the main work:

clustering():
	model = _______( include_top =False, weights="imagenet", input_shape=(224,223,3)) # gets the weights of the pen-ultimate layer and hence the model of ___.
	output = covnet_transform(model, list of images) # returns flattened output out of the model and get's the vectorized output.
	pca = create_fit_PCA(output) # returns the pca output of the moel's output so as to reduce the dmensions further before passing it to the kmeans clustering model.
	k_pca = kMeans(n_clusters, n_jobs=-1, random_state=728) # creates a kmeans clustering model with n_clusters.
	k_pca.fit(output_pca) # fits, ie; trains the modelwith the output of pca ( the reduced dimenions array ).
	pred_pca = k_pca.predict(output_pca) # predicts the value w.r.t. the kmeans clustering model.
	#Then copies files to the approprirate folder using the predicted cluster values

"""


# Imports
import time, random, cv2, glob, keras, os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
from shutil import copy2

# flattens and gives the vectorized output out of the covnet_model passed.
def  covnet_transform(covnet_model, raw_images):
	pred = covnet_model.predict(raw_images)
	flat = pred.reshape(raw_images.shape[0], -1)
	return flat

# reduces the dimensions using pca and returns the new reduced and vectorized out
def create_fit_PCA(data, n_components=None):
    p = PCA(n_components=n_components, random_state=728)
    p.fit(data)
    return p

class image_clustering:
	def __init__(self, path="data", n=10):
		self.data_paths = glob.glob(path + "\*.png")
		random.shuffle(self.data_paths)
		self.n_clusters = n
		os.makedirs("output")
		for i in range(n):
			os.makedirs("output\\vgg16\\cluster" + str(i))
			os.makedirs("output\\vgg19\\cluster" + str(i))
			os.makedirs("output\\resnet50\\cluster" + str(i))
		self.images = []
		for image in self.data_paths:
			self.images.append(cv2.cvtColor(cv2.resize(cv2.imread(image), (224,224)), cv2.COLOR_BGR2RGB))
		self.images = np.float32(self.images)
		self.images /= 255
		print("\n Object of class \"image_clustering\" has been initialized.")
		
	def vgg16_clustering(self):
		print("\n\nVGG16 and KMEANS\n")
		vgg16_model = keras.applications.vgg16.VGG16(include_top=False, weights="imagenet", input_shape=(224,224,3))
		print("\n transforming")
		vgg16_output = covnet_transform(vgg16_model,self.images)
		print("\n transformed")
		vgg16_pca = create_fit_PCA(vgg16_output)
		print("fit")
		K_vgg16_pca = KMeans(n_clusters=self.n_clusters, n_jobs=-1, random_state=728)
		print("\n training starts")
		start = time.time()
		k_vgg16_pca.fit(vgg16_output_pca)
		end = time.time()
		print("\nTraining using vgg16 and kmeans took " + str(end-start) + " seconds.\nPredicting..")
		self.k_vgg16_pred_pca = K_vgg16_pca.predict(vgg16_output_pca)
		for i in range(self.n_examples):
			copy2(self.data_paths[i], "output\\vgg16\\cluster" + str(k_vgg16_pred_pca[i]))
		print("\nImages stored in the approprirate clusters according to vgg16 and kmeans in the output\\vgg16 folder.")

	def vgg19_clustering(self):
		print("\n\nVGG19 and KMEANS\n")
		vgg19_model = keras.applications.vgg19.vgg19(include_top=False, weights="imagenet", input_shape=(224,224,3))
		vgg19_output = covnet_transform(vgg19_model,self.images)
		vgg19_pca = create_fit_PCA(vgg19_output)
		K_vgg19_pca = KMeans(n_clusters=self.n_clusters, n_jobs=-1, random_state=728)
		start = time.time()
		k_vgg19_pca.fit(vgg19_output_pca)
		end = time.time()
		print("\nTraining using vgg19 and Kmeans took " + str(end-start) + " seconds.\nPredicting..")
		self.k_vgg19_pred_pca = K_vgg19_pca.predict(vgg19_output_pca)
		for i in range(self.n_examples):
			copy2(self.data_paths[i], "output\\vgg19\\cluster" + str(k_vgg19_pred_pca[i]))
		print("\nImages stored in the approprirate clusters according to vgg19 and kmeans in the output\\vgg19 folder.")

	def resnet50_clustering(self):
		print("\n\nRESNET50 and KMEANS\n")
		resnet50_model = keras.applications.resnet50.ResNet50(include_top=False, weights="imagenet", input_shape=(224,224,3))
		resnet50_output = covnet_transform(resnet50_model,self.images)
		resnet50_pca = create_fit_PCA(resnet50_output)
		K_resnet50_pca = KMeans(n_clusters=self.n_clusters, n_jobs=-1, random_state=728)
		start = time.time()
		k_resnet50_pca.fit(resnet50_output_pca)
		end = time.time()
		print("\nTraining using resnet50 and Kmeans took " + str(end-start) + " seconds.\nPredicting..")
		self.k_resnet50_pred_pca = K_resnet50_pca.predict(resnet50_output_pca)
		for i in range(self.n_examples):
			copy2(self.data_paths[i], "output\\resnet50\\cluster" + str(k_resnet50_pred_pca[i]))
		print("\nImages stored in the approprirate clusters according to resnet50 and kmeans in the output\\resnet50 folder.")

def main():

	number_of_clusters = 3
	data_path = "data"

	temp = image_clustering(data_path, number_of_clusters)
	temp.vgg16_clustering()
	temp.vgg19_clustering()
	temp.resnet50_clustering()
	print("\n\n\t\t END\n\n")

# Run
main()