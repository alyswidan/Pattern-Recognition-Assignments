This repo has my solutions to the Pattern recognintion course at the Faculty of Engineering Alexandria University.
The course had 7 homework problem sets and 4 assignments.
The course [website](https://sites.google.com/view/ssp-pr-torki/home) has more information about the assigmnets and homework problems. 
## Home work Problems
### PCA 
- Implementing the principal component analysis algorithm.
- Applying the implementation to a small dataset.
- Handwritten problems on the mathimatics of the algorithm.

### LDA
- Implementing the Linear Descriminat analysis algorithm.
- Applying the implementation to a small dataset.
- Handwritten problems on the mathimatics of the algorithm.

### Naive Bayes
- Implementing the Naive Bayes algorithm.
- Applying Naive Bayes and Full Bayes to a toy dataset.

### Decision Trees
- Implementing the Decision Tree algorithm from scratch.
- Applying the Implementation to the [orl faces](www.cl.cam.ac.uk/Research/DTG/attarchive/pub/data/att_faces.tar.Z) data set.
- Using sklearn Random Forests on the same dataset.
- Using PCA with Random Forests and Decision Trees and comparing the results of different procedurs.

### SVM and Linear Regression
- Experimenting with soft margin SVMs on a toy dataset.
- Visualizing resulting decison boundries.
- Using the resulting classifiers in an ensemble to classify the points by hand.
- Using sklearn's Linear Regression on a toy dataset and visualizing the resulting planes for different values of the regularization parameter alpha.


### k-means, spectral clustering and clustering evaluation
- Implementing kmeans and applying it to applying in to a toy dataset.
- Implementing normalized cut spectral clustering with rbf and knn affinity matrices and comparing the plots of the embedding space for both methods.
- Implementing a number of clustering internal and external evaluation measures and comparing the performance of the above clustering measures on the toy dataset using plots.


## Assignments
Every Assignment is contained in its own repo.Links to the repos are included in the corresponding
description below.

### [Face Recognition](https://github.com/alyswidan/FaceRecognition)
- Implemented PCA and multiclass LDA.
- Using EigenFaces and FisherFaces to recognise faces in the [orl faces](www.cl.cam.ac.uk/Research/DTG/attarchive/pub/data/att_faces.tar.Z) data set.

### [IMDB movie reviews sentiment analysis](https://github.com/alyswidan/imdb-sentiment-analysis)
Performing sentiment analysis on the IMDB movie reviews [dataset]()
- Used NLTK to preprocess the reviews.
- Compared different vectorization techniques
    * word2vec using gensim
    * doc2vec with multple variations using gensim
    * TF-IDF using sklearn 
- Compared different traditional non deeplearning classifires.
- Obtained an accuracy of 92.8% using TF-IDF with a logistic regression classifier.

### [Modulation Recognition]()


### [Image Segmentation](https://github.com/alyswidan/Image_Segmentation)
Performing image segmentation on the Berkley Segmentation [dataset](https://sites.google.com/view/ssp-pr-torki/home)
- Tried kMeans with different values of k.
- Tried normalized cut spectral clustering using a k-nearest neighbours affinity matrices.
- Evaluated a compared results using conditional entropy and F-measure.
