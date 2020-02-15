library(igraph)

interactions_combined = read.table("/home/alex/Documents/M2/RESYS/TP/TP2/human_combined_900.txt", header=T)
interactions_experimental = read.table("/home/alex/Documents/M2/RESYS/TP/TP2/human_experimental_900.txt", header=T)

#library(ggplot2)


my_hist = function(data, plot_density=FALSE, transform=NULL, theme=theme_classic, vline=NULL){
  varname = deparse(substitute(data))
  plot = ggplot(data.frame(data=data), aes_string(x=data)) + 
    xlab(varname) +
    geom_histogram(aes(y=..density..), alpha=0.5, fill='lightblue', color='grey') +
    theme()
  if(plot_density) plot = plot + geom_density()
  if(!is.null(transform)) plot = plot + scale_x_continuous(trans=transform)
  if(!is.null(vline)) plot = plot + geom_vline(xintercept = vline, color='orange')
  
  plot
}

g = graph.data.frame(interactions_combined, directed=TRUE, vertices=NULL)
adj_mat <- as_adj(g)

degree = list() 
for(n in adj_mat) {
  degree.append( sum(n) )
}
#my_hist(degree)
