#Installing and loading the required packages and libraries
packages <- c("readxl", "tidyverse","gridExtra","ggcorrplot","dplyr")
#install.packages(packages)

library(readxl)
library(tidyr)
library(tidyverse) 
library(ggplot2)   
library(gridExtra)
library(ggcorrplot)
library(rlang) 
library(factoextra)
library(NbClust)
library(cluster)

# read the dataset 
CW_Winedata <- read_excel('G:\\iit campus\\course\\2 ND YEAR\\2 sem\\Machine Learning and Data mining\\cw\\Whitewine_v6.xlsx')
# remove 12th column
CW_Winedata <- CW_Winedata[,1:11]
# scale the dataset
CW_Winedata <- scale(CW_Winedata)
# check if any missing values
colSums(is.na(CW_Winedata))

# calculate the Covariance Matrix
cw.cov <- cov(CW_Winedata)
ggcorrplot(cw.cov)


# Apply the built-in prcomp function to get results
CW_pca <- prcomp(CW_Winedata)
names(CW_pca)
summary(CW_pca)
# eigen values
CW_pca$sdev^2
# Principal Component Loadings/ eigen vectors
CW_pca$rotation
CW_pca$rotation <- -CW_pca$rotation
# calculate cumulative score of PCs
c_score <- cumsum(CW_pca$sdev^2 / sum(CW_pca$sdev^2))
c_score
# given that cumulative score is from 85% is selected  
CW_pca_new <- which(c_score < 0.85)
CW_pca_new
# new transformed dataset with selected principal components
CW_pca_transform = as.data.frame(-CW_pca$x[,1:7]) 
CW_pca_transform

# Disscussion---
# scree plot 
fviz_eig(CW_pca, addlabels = TRUE)
# variables of the PCA
fviz_pca_var(CW_pca, col.var = "blue")
# barplot 
var_precent <- (CW_pca$sdev^2 / sum(CW_pca$sdev^2))*100
barplot(var_precent, xlab='PC', ylab='Percent Variance', 
        names.arg=1:length(var_precent), 
        las=1, ylim=c(0, max(var_precent)), 
        col='gray')
# biplot
biplot(CW_pca, scale = 0)

#--------- Determine the number of cluster centres
set.seed(10)
# elbow method 
fviz_nbclust(CW_pca_transform, kmeans, method = "wss") +
  labs(subtitle = "Elbow method")
set.seed(10)
# Silhouette method
fviz_nbclust(CW_pca_transform, kmeans, method = "silhouette")+
  labs(subtitle = "Silhouette method")
set.seed(10)
# Gap statistic method
fviz_nbclust(CW_pca_transform, kmeans,  method = "gap_stat")+
  labs(subtitle = "Gap statistic method")
set.seed(10)
# NBclust method
NbClust(CW_pca_transform, distance="euclidean", min.nc=2, max.nc=10, method="kmeans", index="all")
NbClust(CW_pca_transform, distance="manhattan", min.nc=2, max.nc=10, method="kmeans", index="all")
NbClust(CW_pca_transform, distance="maximum", min.nc=2, max.nc=10, method="kmeans", index="all")

# kmeans
kmeans_data <- kmeans(CW_pca_transform, centers = 2, nstart = 20)
# display the structure of object
str(kmeans_data)
# print the object
kmeans_data
# display clusters
fviz_cluster(kmeans_data, data = CW_pca_transform)
# display clusters
fviz_cluster(kmeans_data, data = CW_pca_transform, ellipse.type = "euclid",star.plot = TRUE, repel = TRUE, ggtheme = theme_minimal())


# get BSS and TSS from kmeans_data object
(BSS <- kmeans_data$betweenss) 
(TSS <- kmeans_data$totss)
(WSS <- TSS - BSS)
# Calculate the ratio of BSS over TSS
(BSS_TSS_ratio <- BSS / TSS * 100)

# silhouette plot analyis 
set.seed(10)
# run kmeans  
km_data <- kmeans(CW_pca_transform, 2, nstart = 10)
fviz_cluster(km_data, CW_pca_transform)
# calculate silhouette index
sil_data <- silhouette(km_data$cluster, dist(CW_pca_transform))
fviz_silhouette(sil_data)
# negative silhouette width
neg_sil_data <- which(sil_data[, "sil_width"] < 0)
sil_data[neg_sil_data, , drop = FALSE]
# average silhouette information
avg_sil_width <- round(mean(sil_data[, 3]),2)
# Evaluate the quality of the clustering
if (avg_sil_width > 0.7) {
  print("The clustering is reasonably good.")
} else if (avg_sil_width >= 0.25) {
  print("The clustering is fair.")
} else {
  print("The clustering is poor.")
}


# Calculate Calinski-Harabasz Index
CH_metric <- function(cluster_obj, data) {
  num_clusters <- length(unique(cluster_obj$cluster))
  num_Rows <- nrow(data)
  BSS <- cluster_obj$betweenss
  WSS <- cluster_obj$tot.withinss
  CH_index <- ((num_Rows - num_clusters) / (num_clusters - 1)) * (BSS / WSS)
  return(CH_index)
}
ch_index <- CH_metric(km_data, CW_pca_transform)
ch_index

# Function to calculate CH index for a given number of clusters
CH_metric <- function(data, num_clusters) {
  cluster_obj <- kmeans(data, centers = num_clusters)
  num_Rows <- nrow(data)
  BSS <- cluster_obj$betweenss
  WSS <- cluster_obj$tot.withinss
  CH_index <- ((num_Rows - num_clusters) / (num_clusters - 1)) * (BSS / WSS)
  return(CH_index)
}

# Function to calculate CH index for a range of cluster numbers
ch_values <- function(data, kmax) {
  ch_indices <- sapply(2:kmax, function(k) {
    CH_metric(data, k)
  })
  return(ch_indices)
}

# Calculate CH indices
ch_indices <- ch_values(CW_pca_transform, 10)  

# Plot CH indices
barplot(ch_indices, names.arg = 2:10, xlab = "Number of clusters", ylab = "CH index",
        main = "CH index for different numbers of clusters", col = "lightblue")