# Install and load required packages
packages <- c("readxl", "tidyverse","factoextra",  "NbClust")
# install.packages(packages)
library(readxl)
library(tidyverse)
library(tidyr)
# read the dataset 
CW_Winedata <- read_excel('G:\\iit campus\\course\\2 ND YEAR\\2 sem\\Machine Learning and Data mining\\cw\\Whitewine_v6.xlsx')
# remove 12th column
CW_Winedata <- CW_Winedata[,1:11]
# check the dimensions
dim(CW_Winedata)
# get first 10 lines of the dataset
head(CW_Winedata,10)
# Before outlier removal
boxplot(CW_Winedata)


# -------Outlier removal with Interquartile Range method
# Define a function to remove outliers using the Interquartile Range method
Cw_out_iqr <- function(x) {
  q1 <- quantile(x, 0.25)
  q3 <- quantile(x, 0.75)
  iqr <- q3 - q1
  lower_bound <- q1 - 1.5 * iqr
  upper_bound <- q3 + 1.5 * iqr
  x[which(x < lower_bound | x > upper_bound)] <- NA
  return(x)
}

# Apply the function to each numerical column in the dataset
CW_Winedata <- as.data.frame(apply(CW_Winedata, 2, Cw_out_iqr))

# Remove rows with NA values
newCW_Winedata <- drop_na(CW_Winedata)

# Visualize boxplots after removing outliers
boxplot(newCW_Winedata)

# ------- Outlier removal using the box plot's statistical properties
CW_Winedata <- scale(CW_Winedata)
col = c('fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol')
for (columnY in col)
{
  value = CW_Winedata[,columnY][CW_Winedata[,columnY] %in% boxplot.stats(CW_Winedata[,columnY])$out]
  CW_Winedata[,columnY][CW_Winedata[,columnY] %in% value] = NA
}

as.data.frame(colSums(is.na(CW_Winedata)))

newCW_Winedata <- as.data.frame(CW_Winedata)
newCW_Winedata <- drop_na(newCW_Winedata)
as.data.frame(colSums(is.na(newCW_Winedata)))
boxplot(newCW_Winedata)


# scale if used iqr method 
newCW_Winedata <- scale(newCW_Winedata)
boxplot(newCW_Winedata)

#--------- Determine the number of cluster centres
library(NbClust)
library(factoextra)
set.seed(10)
# elbow method 
fviz_nbclust(newCW_Winedata, kmeans, method = "wss") +
  labs(subtitle = "Elbow method")
set.seed(10)
# Silhouette method
fviz_nbclust(newCW_Winedata, kmeans, method = "silhouette")+
  labs(subtitle = "Silhouette method")
set.seed(10)
# Gap statistic method
fviz_nbclust(newCW_Winedata, kmeans,  method = "gap_stat")+
  labs(subtitle = "Gap statistic method")
set.seed(10)
# NBclust method
NbClust(newCW_Winedata, distance="euclidean", min.nc=2, max.nc=10, method="kmeans", index="all")
NbClust(newCW_Winedata, distance="manhattan", min.nc=2, max.nc=10, method="kmeans", index="all")
NbClust(newCW_Winedata, distance="maximum", min.nc=2, max.nc=10, method="kmeans", index="all")

# --- K- means clustering
kmeans_data <- kmeans(newCW_Winedata, centers = 2, nstart = 20)
# display the structure of object
str(kmeans_data)
# print the object
kmeans_data
# display clusters
fviz_cluster(kmeans_data, data = newCW_Winedata)
# display clusters
fviz_cluster(kmeans_data, data = newCW_Winedata, ellipse.type = "euclid",star.plot = TRUE, repel = TRUE, ggtheme = theme_minimal())


# get BSS and TSS from kmeans_data object
(BSS <- kmeans_data$betweenss)   #  5737.309
(TSS <- kmeans_data$totss)       # 24662
(WSS <- TSS - BSS)               # 18924.69
# Calculate the ratio of BSS over TSS
(BSS_TSS_ratio <- BSS / TSS * 100)  # 23.26376%

# Create a data frame
df <- data.frame(
  Index = c("BSS", "WSS"),
  Value = c(BSS, WSS)
)

# Create a bar plot
library(ggplot2)
ggplot(df, aes(x=Index, y=Value, fill=Index)) +
  geom_bar(stat="identity") +
  theme_minimal() +
  labs(x="Index", y="Value", title="BSS and WSS Indices", fill="Index")

# silhouette plot analysis 
library("cluster")
set.seed(10)
# run kmeans  
km_Resultsdata <- kmeans(newCW_Winedata, 2, nstart = 10)
fviz_cluster(km_Resultsdata, newCW_Winedata)
# calculate silhouette index
sil_data <- silhouette(km_Resultsdata$cluster, dist(newCW_Winedata))
fviz_silhouette(sil_data)
# negative silhouette width
neg_sil_data <- which(sil_data[, "sil_width"] < 0)
sil_data[neg_sil_data, , drop = FALSE]
# average silhouette information
avg_sil_width <- mean(sil_data[, 3])
# Evaluate the quality of the clustering
if (avg_sil_width > 0.7) {
  print("The clustering is reasonably good.")
} else if (avg_sil_width > 0.25) {
  print("The clustering is fair.")
} else {
  print("The clustering is poor.")
}

# run the Partitioning Around Medoids 
pam.data <- pam(newCW_Winedata, 2)
fviz_cluster(pam.data)
fviz_silhouette(pam.data)


