install.packages('Seurat')
library(Seurat)

library(MASS) # Various statistical methods
library(ppcor) # Partial correlation
library(parmigene) # ARACNE, CLR implementation with mutual information
library(infotheo) # Information theory measures
library(miic) # Causal graph inference with mutual information
library(dplyr) # Data processing
library(knitr) # Markdown
library(ggplot2) # Plots
library(gridExtra) # Plots extra
library(Seurat) # Single cell data manipulation
#source("utils.R")
source("./utils.R")


install.packages("cowplot")
install.packages("UMAPPlot")


source("https://bioconductor.org/biocLite.R")

if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install()

BiocManager::install("GENIE3")
biocLite("GENIE3")





rm(list = ls()) #clear environment

load("/home/alex/Documents/RESYS/TP3/single_cell_data.Rdata")
#load("/home/alex/Documents/RESYS/TP3/single_cell_data_small.Rdata")
ls()
