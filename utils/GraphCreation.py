import numpy as np
import random
import matplotlib.pyplot as plt
import networkx as nx

def graphCreation(n,M,k,clusters_size,prob_list,prob_list_prime, plot_adj = False, plot_graph = False):
  """
  Parameters
  ----------
  n: number of nodes
  M: number of layers
  k: number of clusters
  clusters_size: list of size k containing the number of nodes in each cluster
  prob_list: list of size k containing the probability of connection inside each cluster
  prob_list_prime: list of size k*k containing the probability of connection between each cluster

  plot_adj: if True, plot the adjacency matrix of each layer
  plot_graph: if True, plot the graph of each layer

  Returns
  -------
  res: list of size M containing the adjacency matrix of each graph
  index_cluster: list of size n containing the cluster of each node
  """

  if(len(clusters_size)!=k or prob_list.shape !=(k,M) or prob_list_prime.shape !=(k,k,M)):
    print("Error in cluster creation ")
    return 0

  index_cluster = []
  for i in range(k):
    index_cluster += clusters_size[i]*[i]
  index_cluster = np.array(index_cluster)

  res = np.zeros((n,n,M))
  for i in range(M):
    for j in range(n):
      for k in range(j):
        if(index_cluster[j]==index_cluster[k]):
          res[j,k,i] = 1 if random.random() < prob_list[index_cluster[j],i] else 0
        else:
          res[j,k,i] = 1 if random.random() < prob_list_prime[index_cluster[j],index_cluster[k],i] else 0
        res[k,j,i] = res[j,k,i]

  # Plot the M adjacency matrices
  if(plot_adj):
    plt.figure(figsize=(15, 3.5*(M//5 + 1)))
    for i in range(M):
      plt.subplot(M//5 + 1, min(5, M), i+1)
      x,y = np.where(res[:,:,i] == 1)
      plt.scatter(x,y, s = 1)
      plt.title("Adjacency matrix of layer " +str(i+1))

    plt.tight_layout()
    plt.show()

  # Plot the M layers 
  if(plot_graph):
    fig, axs = plt.subplots(2, 1+M//2, figsize=(12, 12)) 

    G = nx.Graph()
    list_nodes = [i for i in range(n)]
    G.add_nodes_from(list_nodes)

    for i in range(n):
      for j in range(i+1,n):
        if(res[i,j,0] == 1):
          G.add_edge(list_nodes[i],list_nodes[j])


    node_colors = index_cluster
    clusters = index_cluster


    pos = nx.spring_layout(G)
    nx.draw(G, pos,ax=axs[0,0], with_labels=True, node_size=300, node_color=node_colors, font_size=10, font_color="black")
    for p in range(1,M):
      G = nx.Graph()
      G.add_nodes_from(list_nodes)

      for i in range(n):
        for j in range(i+1,n):
          if(res[i,j,p] == 1):
            G.add_edge(list_nodes[i],list_nodes[j])



      nx.draw(G, pos,ax=axs[p%2, p//2], with_labels=True, node_size=300, node_color=node_colors, font_size=10, font_color="black")


    axs[1, 1].axis('off')
    plt.tight_layout()
    plt.plot()

  return res,index_cluster