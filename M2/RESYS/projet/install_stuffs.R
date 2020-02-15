rm(list = ls()) #clear environment
update.packages()

#if (!requireNamespace("BiocManager", quietly = TRUE))
#  install.packages("BiocManager")
#BiocManager::install()
#BiocManager::install("GENIE3")
#BiocManager::install("Rgraphviz")

#install.packages("funModeling")

remove.packages("rstan")
if (file.exists(".RData")) file.remove(".RData")
Sys.setenv(MAKEFLAGS = "-j2") # four cores used
install.packages("rstan", type = "source")


dotR <- file.path(Sys.getenv("HOME"), ".R")
if (!file.exists(dotR)) dir.create(dotR)
M <- file.path(dotR, "Makevars.win")
if (!file.exists(M)) file.create(M)
cat("\nCXX14FLAGS=-O3 -march=native",
    "CXX14 = g++ -m$(WIN) -std=c++1y",
    "CXX11FLAGS=-O3 -march=native",
    file = M, sep = "\n", append = TRUE)


remove.packages("miic", "C:/Users/Alex/Documents/R/win-library/3.6" )


install.packages("http://miic.curie.fr/download/miic_mixed.tar.gz", repos = NULL, type = "source")

# Install MIIC from remote private repository
if (!require(miic)) install.packages(
  "https://miic.curie.fr/download/miic_mixed.tar.gz", 
  repos = NULL, type = "source"
)

library(miic) # MIIC package



install.packages('Seurat', dependencies= TRUE)
