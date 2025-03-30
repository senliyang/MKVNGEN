# MKVNGEN:Predicting Microbe-Disease Associations Based on Multi-Kernel Autoencoder and Ensemble Learning
Ample research evidence suggests that many complex diseases in humans are associated with microbial communities. Therefore, identifying potential associations between microbes and diseases holds significant importance for disease diagnosis, prognosis, and treatment. However, traditional biomedical experiments are costly, time-consuming, and labor-intensive. Thus, we propose a novel computational model, MKVNGEN, to predict potential associations between microbes and diseases. First, we leverage a multi-kernel learning algorithm based on the Hilbert-Schmidt Independence Criterion (HSIC-MKL) to fuse multi-source features of microbes and diseases based on known microbe-diseases associations. Next, the model predicts potential microbe-disease associations using feature extraction from Variational Graph Autoencoder, non-negative matrix factorization, Graph Attention Autoencoder, combined with deep neural networks and Interpretable Boosting Machines for classification to predict potential microbes-disease associations.  Through 5-fold cross-validation, we compare MKVNGEN with five advanced predictive methods and six classical ensemble learning classifiers. Experimental results demonstrate that the performance of MKVNGEN is significantly superior to other methods using the HMDIP and HMDAD databases. Furthermore, in case studies of Parkinson's disease, obesity, Crohn's disease, and colorectal cancer, most of the microbes predicted to be associated with these diseases have been validated, further confirming the reliability of MKVNGEN in predicting microbe-disease associations.MKVNGEN is available at https://github.com/senliyang/MKVNGEN.
# Flowchart
![image](https://github.com/senliyang/MKVNGEN/blob/main/%E6%A8%A1%E5%9E%8B%E6%A1%86%E6%9E%B6.png)
# Requirements
Install python3.9 for running this model. And these packages should be satisfied:
tensorflow-gpu ≈2.6.0
numpy ≈ 1.19.5
pandas ≈ 1.1.5
scikit-learn ≈ 0.24.2
# Usage
Default is 5-fold cross validation. To run this model：python main.py
Calculate the integrated similarity between microbes and diseases  　&ensp;                  python generate_kernel.py          
Extract the linear features of microbes and diseases             　&ensp;        python NMF.py                 
Extract the nonlinear features of microbes and diseases          　&ensp;      python GATE.py ;python VGAE.py
