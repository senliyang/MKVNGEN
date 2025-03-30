# MKVNGEN
# Flowchart
![image][(https://github.com/senliyang/MKVNGEN/blob/main/model.png)]
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
