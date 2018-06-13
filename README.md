# image-clustering

### This is a simple unsupervised image clustering algorithm which uses KMeans for clustering and Keras applications with weights pre-trained on ImageNet for vectorization of the images.

A folder named "output" will be created and the different clusters formed using the different algorithms will be present. 

Change the following variables(present in the main() function) as per your convinience:
1) number_of_clusters - The number of clusters to be created by the clustering algorithm. (default is 10)
2) data_path - This is the path of the folder that contans the different images that you want to pass to the algorithm.
3) max_examples - The max number of examples to be used for the clustering (if None, all the images in the data_path folder will be used)
4) use_imagenets - choose which keras application to use. (Choose from: "Xception", "VGG16", "VGG19", "ResNet50", "InceptionV3", "InceptionResNetV2", "DenseNet", "MobileNetV2" and "False". If False, the image will be passed as is to the clustering algorithm)
5) use_pca - choose whether to use PCA for dimentionality reduction. (choose betwwen between "True" and "False". If use_imagenets=False, then use_pca will automatically be set to False as well) 

### Python Modules used:  
-Keras 
-Theano (as backend for keras)  
-os
-random  
-cv2 (openCV)  
-numpy  
-sklearn (scikit-learn for KMeans and PCA)  
-shutil

### To run: "python image_clustering.py"
