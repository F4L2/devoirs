# https://drive.google.com/drive/folders/18LzbHb3Dcf43h7dXD334ZAXbrsQ8sIiD


install.packages("igraph", dependencies= TRUE)


library(igraph)

g = read_graph("/home/alex/Documents/M2/RESYS/TP/TP1/simple_graph.txt")
g = as.undirected(g)
plot(g)

