
# AI Application System 


````
Name: Akash Ghimire
student id: 12194814

````

>**For each week task respective folders have been created. Each folders contains the jupyter notebook files that we practised during the class. If there were more than one coding session in a single week then that respective week folders will have multiple jupyter notebook files.** 


<br>

## Week 3_and_4
In these weeks we learn most of the theoretical stuffs behind how simple preceptons in deep learning networks works. **We learned about basics component of preceptron i.e.weights and bias. We also learn about ** forward passing, activation function, backward passing, loss , and error functions.** 
<br>
> In the lab section we build simple neural network from the scratch. The code can be found under week 3_and_4 folder. The codes are explained well using jupyter notebook markdown language. 



  

## Week 5
In this week we learned basic of tensors in tensorflow which including creating tensors and performing different mathematical operation with created tensors. Furthermore, we learned about computational graph and visualizing our task using tensorboard to analyse and visualize our task. 
To make the task simpler i have created separate jupyter notebook for each of these smaller tasks. In this week_folder i have also added some extra jupyter notebook files which helped me to learn about basics of tensorflow further more.
<br>



## Week 6
In this week we have two lab session. In the *first lab session* we learn to train custom made dataset using tensorflow. The training include implementation of a simple linear regression algorithm.  
In the first lab session of this week we learned the following: 1)  training a simple **Linear Regression** model on custom training datasets and 2) training a model on **MNIST**

In the *second lab session*, we learned to denoised the image using Deep learning technique known as ***autoencoder***.
<br>
> **An autoencoder is a type of artificial neural network used to learn efficient codings of unlabeled data (unsupervised learning).[1] The encoding is validated and refined by attempting to regenerate the input from the encoding. The autoencoder learns a representation (encoding) for a set of data, typically for dimensionality reduction, by training the network to ignore insignificant data (“noise”).**



## Week 7
In this week we learned about theoretical and coding aspects of Convulutional Neural Network(CNN). During the lab session we used cifar 10 datasets and trained our CNN model on it. To visualise the result we use tensorboard callback.
<br>
>**A convolutional neural network (CNN) is a deep learning network architecture that learns directly from data, eliminating the need for manual feature extraction.CNNs are especially useful for detecting patterns in images in order to recognize objects, faces, and scenes.**

<br>


> The CIFAR-10 dataset (Canadian Institute for Advanced Research, 10 classes) is a subset of the Tiny Images dataset and consists of 60000 32x32 color images. The images are labelled with one of 10 mutually exclusive classes: airplane, automobile (but not truck or pickup truck), bird, cat, deer, dog, frog, horse, ship, and truck (but not pickup truck). There are 6000 images per class with 5000 training and 1000 testing images per class.


<br>


## Week 9

In the first session of this week we learn to predict the housing price using linear regression. For this we use famous **Boston Housing** Dataset.

In the second session of this week, we learn how to handle overfitting during training of dataset. We learned about **Regularization, L2 and Dropout Layer** and implemented them in the code aswell. At the end we analyse the effective of Dropout and Regularization on overfitting and training problem.
 
<br>

## Week 10
In this week we learned about **transfer learning** and image classification using CNN with transfer learning. 
<br>
In the first session, we learn to use transfer learning on **pytorch** framework. For this purpose we used ResNet backbone to predict the image of a dog. 

In the second and third session, we learn how to use *sequential and functional* API in **tensorflow** framework.

<br>

## Week 11
In this week we learn about recurrent neural network (RNN).
> RNN: A recurrent neural network is a class of artificial neural networks where connections between nodes can create a cycle, allowing output from some nodes to affect subsequent input to the same nodes. This allows it to exhibit temporal dynamic behavior. (src: wikipedia)

In this sesssion, we predicted the sales of books using RNN network. 

<br>

## Week 12

In this week we learn about **Weather forecast using RNN, LSTM.**
<br>
Furthermore, in the coding session we also learned things like how we can build simple RNN using numpy. We learn **how to use LSTM,GRU, and Bidirectional LSTM using keras.**

<br>


## Week 13

In this week we learned about Natural Language Processing (NLP).
> Natural language processing is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data. (src:Wikipedia)

In this week, we learn how words are reprensented in machine learning. We learn about important aspect of NLP such as  **word embedding, text vectorization, One hot encodding.**  
<br>
We had two session of coding. In both of these session we used RNN based model and codes were implemented in **tensorflow** framework .

<br>

## Week 14 
Like week 13, in this week as well we did activites related to NLP but instead of uisng RNN based model like LSTM/GRU we used **Transformer Encoder**. 
<br>
Some Advantages of using transformer instead of RNN models are:
1. It does not suffers from vanishing gradient like RNN model.
2. It can handle long term dependency. 
3. Transformer Encoder are better suited for transfer learning compared to LSTM/GRU models.
