# Multilayer-Graph-Clustering

TODO 

- chercher state of the art pour une couche 
- quand on moyenne c'est pas vrai que leur algo est meilleur
- Expériences avec toutes les mêmes probas
    - Pour chaque expérience : courbe selon M, valeurs probas

- KKmeans et SC_SR à intégrer (on peut commencer déjà les expérimentations, ça prendra 2 sec de les rajouter quand on les aura, cf le tableau algorithms dans Run.ipynb)
    - Meilame: j'ai peut être compris pour le SC_SR M layers, y a une histoire de faire un spectal clustering classique sur CHAQUE layer en solo, compute la NMI par rapport au true_clusters et ça nous donne le classement des layers les plus informatives. Ensuite algo glouton (je regarderai ça semaine pro)

- Expérimentations avec différents types de matrices... faire varier k, M, n, delta... essayer de voir quand SC_GED marche ou quand au contraire il sert à rien vs les baselines
- LATEX
- Max: check le "variant SC_GED" dans [A SUPPRIMER]SC_GE.ipynb, je sais pas trop ce que c'est... éventuellement à mettre dans utils/Algorithms et supprimer le notebook



----------STRUCTURE DU RAPPORT-------------

0. Abstract
1. Intro and problem statement
2. Algorithms
3. Synthetic data generation
4. Results and 
5. Conclusion 



