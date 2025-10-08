# Machine Learning and Deep Learning Projects

This repository contains a collection of machine learning and deep learning projects implemented in Jupyter Notebooks. Each project is self-contained and includes the necessary code to reproduce the results.

---

## Projects

### 1. MNIST Digit Classification with PyTorch

This project builds and trains a Convolutional Neural Network (CNN) to classify handwritten digits from the famous MNIST dataset. It's a great introduction to computer vision using PyTorch.

**Technologies Used:**

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)

**Notebook:** [`Copy_of_PT_Part1_MNIST.ipynb`](./Copy_of_PT_Part1_MNIST.ipynb)

**About the Project:**

The notebook walks through the process of:
1.  Loading and visualizing the MNIST dataset.
2.  Building a simple fully connected neural network.
3.  Building a more advanced Convolutional Neural Network (CNN).
4.  Training and evaluating both models.

**Model Architecture:**

*Fully Connected Network:*
<img src="https://raw.githubusercontent.com/MITDeepLearning/introtodeeplearning/master/lab2/img/mnist_2layers_arch.png" width="400"/>

*Convolutional Neural Network:*
<img src="https://raw.githubusercontent.com/MITDeepLearning/introtodeeplearning/master/lab2/img/convnet_fig.png" width="600"/>

---

### 2. Genomic Sequence Generation with an LSTM

This project uses a Long Short-Term Memory (LSTM) network to learn patterns from the Lassa virus genome and generate new, synthetic genomic sequences.

**Technologies Used:**

![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)

**Notebook:** [`genomics-1.ipynb`](./genomics-1.ipynb)

**About the Project:**

This notebook covers the following steps:
1.  **Data Retrieval:** Fetches Lassa virus genome data from the NCBI database using BioPython.
2.  **Preprocessing:** Cleans and prepares the genomic sequences for the model.
3.  **Model Building:** Creates an LSTM model with TensorFlow and Keras to learn the sequence patterns.
4.  **Training and Evaluation:** Trains the model and visualizes the training history (accuracy and loss). The notebook generates plots for these metrics.

---

Feel free to explore the notebooks and reach out with any questions!