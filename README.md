# MachineLearning-ImageSegmentationCNN
The second assignement done for the CS342 Machine Learning module at Warwick University.

The Assignment was called "Kaggle Competition: 2018 Data Science Bowl", the full specification of the assignment can be seen at the file "cs342_ass2.pdf", which is suggested to be read in order to underdstand the problem.

The Goal of this project was to learn some advanced machine learning tecniques in the field of image processing and segmentation, including applying Neural Networks and research level deep Learning arichtectures to take part in a public Kaggle competition using libriaries like keras and tensorflow.

By working through this project I was able ,among other things, to learn the theory and the practice of:

- Basic image-specific exploration tools
- Preprocessing and morphological segementation techniques:
  - Histogram of gradients
  - Watershed segmentation
  - Sobel gradient
  - otsu masking
- Feature engineering and data augmentation
- Multi Layer Perceptron
- Convolutional Neural Networks
  -convolution
  -max-pooling and other methods to avoid overfitting
  -different models and architectures
  

The file 'ReportML(1).pdf' contains my final report for the assignment together with the scores obtained in the kaggle private leaderbord while the files 'prep_and_noNN_pred.py', 'mlp.py' and 'unet.py' are the actual submitted files for the assignment.
prep_and_noNN_pred.py: Preprocessing + attempt at naive segmentation without Neural Networks
mlp.py                : attempt at using a basic multilayer perceptron ( very bad, as expected)
unet.py               : Uses a common CNN architecture found in the literature and used massively used on Kaggle to do the segmentation

Much of the preparation work can be seen in the jupyter notebook 'Master Assignment 2.ipynb' (THIS NEED CLEANING,VERY ROUGH CODE)

