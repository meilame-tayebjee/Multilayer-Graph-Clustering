# Multilayer Graph Clustering
## Max Kramkimel & Meilame Tayebjee

This project has been realised for the Geometric Data Analysis course of the MVA master. by Prof. Jean Feydy.

Its aim is to summarize and discuss the content of the article _Clustering with Multi-Layer Graphs: A Spectral Perspective_ by Xiaowen Dong, Pascal Frossard,
Pierre Vandergheynst and Nikolai Nefedov (the article is available in the repo).

More precisely, we :
- recall the algorithms involved and the results obtained by the authors.
- have a discussion on the hyperparameter selections for the SC-GED and SC-SR algorithms, on the choice of the first layer in the SC-SR, and propose several inference for the M-layer version of SC-SR.
- generate synthetic data and conduct several experiments to assess the benefits and limitations of SC-GED and SC-SR, especially in asymptotic/limit cases.

Everything is detail in the project report also available in the repository. All of the code here is directly runnable and reproduce the results presented.


- Run.ipynb contains all of our main experiments, organized in sections.
- the folders utils contains:
      - Algorithms.py : all of the multilayer graph clustering algorithms involved (SC-SR and affiliated, SC-GED, SC-AL, SC-SUM...)
      - GraphCreation.py : our model for synthetic data generation
      - Metrics.py : the metrics used to evaluate the clustering
      - utils.py : contains several useful methods to process the multilayer graphs as well as the basic Normalized Spectral Clustering algorithm

Additionnaly, the repo also contains supplementary materials:
- SC-GED-parameters.ipynb contains a cross validation for the hyperparameters $\alpha$ and $\beta$ in a specific case
- SC-SR-Supplementary-tests.ipynb re-run every experiments of Run.ipynb but comparing SC-SR, ourSC-SR and randomSC-SR

Fell free to reach out to us at meilame.tayebjee@polytechnique.edu / max.kramkimel@gmail.com for any questions or recommendations.


