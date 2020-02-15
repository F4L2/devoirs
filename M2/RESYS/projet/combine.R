rm(list = ls()) #clear environment

library(miic) # Learning Causal or Non-Causal Graphical Models Using Information Theory
load("/home/alex/Documents/RESYS/projet/miic_graphe2.Rdata")
load("/home/alex/Documents/RESYS/projet/miic_graphe3.Rdata")
load("/home/alex/Documents/RESYS/projet/miic_graphe4.Rdata")
load("/home/alex/Documents/RESYS/projet/miic_graphe5.Rdata")

load("/home/alex/Documents/RESYS/projet/miic_graphe1.RData")
miic_res1 = miic_res

miic_results = c(miic_res1, miic_res2, miic_res3, miic_res4, miic_res5)



miic.plot(miic_res1, igraphLayout = layout_nicely)
miic.plot(miic_res2, igraphLayout = layout_nicely)
miic.plot(miic_res3, igraphLayout = layout_nicely)
miic.plot(miic_res4, igraphLayout = layout_nicely)
miic.plot(miic_res5, igraphLayout = layout_nicely)

save.image("/home/alex/Documents/RESYS/projet/miic_allgraphs.Rdata")





g <- make_graph("Zachary")
sg <- cluster_spinglass(g)
le <- cluster_leading_eigen(g)
compare(sg, le, method="rand")
compare(membership(sg), membership(le))
