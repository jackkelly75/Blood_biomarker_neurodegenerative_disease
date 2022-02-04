library(GEOquery)
library(hgu133plus2.db)
library(limma)
library(R.utils)
library(tidyr)
library(RColorBrewer)
library(GEOquery)
library(affy)
library(ggplot2)


##################################
# 	sort out the phenodata 						
##################################
setwd("C:/Users/Jack/Desktop/Deep_learning/PD_dataset/1 import data")

gse99039 <- getGEO("GSE99039", GSEMatrix = TRUE)
pData <- pData(phenoData(gse99039[[1]]))[,c(1,34,48:54, 56, 57, 60:64)]
pData$fileName <- NA
for (i in 1:nrow(pData)){
  pData$fileName[i] <- paste(rownames(pData)[i], "_", pData$description[i], sep = "")
  pData$fileName[i] <- gsub("\\(|\\)", "", pData$fileName[i])
  pData$fileName[i] <- gsub("Plus_2.CEL", "Plus_2_.CEL", pData$fileName[i])
}

HCpData <- pData[pData$`disease label:ch1`  == "CONTROL",]
PDpData <- pData[pData$`disease label:ch1`  == "IPD",]
pData <- rbind(HCpData, PDpData)
gse99039_pData <- pData


################################
#         Import data          #
################################
#untar("GSE99039_RAW.tar")
#cels = list.files(pattern = "CEL")
#sapply(paste(cels, sep= "/"), gunzip)
cels = gse99039_pData$fileName
unfiltered.raw.data = ReadAffy(filenames = cels, phenoData = gse99039_pData)


################################
#   Plot to detect outliers    #
################################
#This is done differently to the AD datasets as they are different data platforms
setwd("C:/Users/Jack/Desktop/Deep_learning/PD_dataset/1 import data/outlier_plots_before")

###################
#	Age plots
###################
x <- gse99039_pData[order(as.numeric(gse99039_pData$`age at exam:ch1`)),]
png('Sample_age_GSE99039.png', width = 958, height = 614)
plot(x$`age at exam:ch1`, col = factor(x$`disease label:ch1`), xlab = "", ylab = "age", main="Plot of ages of samples GSE99039 (197 are NA)", xaxt='n')
legend(1, 80, legend=c("CTL", "PD"), pch =1,  col=c("black", "red"))
dev.off()

###################
#	Density Plots
###################
brewer.cols <- brewer.pal(9, "Set1")
png("Unprocessed log scale probe intensities.png", height = 1200, width = 1800)
par(mar=c(5.1,5.1,4.1,2.1))
boxplot(unfiltered.raw.data, col = brewer.cols, ylab = "Unprocessed log (base 2) scale Probe Intensities", xlab = "Array Names", main = "Unprocessed log (base 2) scale Probe Intensities of each Array", xaxt='n', cex.main=3.5, cex.lab=3, cex.axis=2)
dev.off()
#Contruct density plots
png("Unprocessed density plots.png", height = 1200, width = 1800)
par(mar=c(5.1,5.1,4.1,2.1))
hist(unfiltered.raw.data, main = "Density plot of log(2) probe intensities", col = brewer.cols, lty=1, xlab = "Log (base 2) Intensities", lwd =3,  cex.main=3.5, cex.lab=3, cex.axis=2)
dev.off()


########################################
#getting the names of sample outliers  #
########################################
png("Outlier Unprocessed density plots.png", height = 1200, width = 1800)
par(mar=c(5.1,5.1,4.1,2.1))
hist(unfiltered.raw.data[,c(62, 167, 296)], main = "Density plot of log(2) probe intensities", col = brewer.cols, lty=1, xlab = "Log (base 2) Intensities", lwd =3,  cex.main=3.5, cex.lab=3, cex.axis=2)
dev.off()

rownames(pData(unfiltered.raw.data))[c(62, 167, 296)]
# "GSM2630912" "GSM2631142" "GSM2630938"
#also remove GSM2630908 as has PINK1 mutation


###############################
#   Remove outliers           #
###############################
remove = c("GSM2630908", "GSM2630938", "GSM2631142", "GSM2630912")
gse99039_pData <- gse99039_pData[!rownames(gse99039_pData) %in% remove, ]

HCpData <- gse99039_pData[gse99039_pData$`disease label:ch1`  == "CONTROL",]
#nrow(HCpData)  #230
PDpData <- gse99039_pData[gse99039_pData$`disease label:ch1`  == "IPD",]
#nrow(PDpData)  #204
gse99039_pData <- rbind(HCpData, PDpData)


################################
#read in the CEL data files    #
################################
setwd("C:/Users/Jack/Desktop/Deep_learning/PD_dataset/1 import data")
cels = gse99039_pData$fileName
unfiltered.raw.data = ReadAffy(filenames = cels, phenoData = gse99039_pData)


################################
#   Plot with outliers removed #
################################
setwd("C:/Users/Jack/Desktop/Deep_learning/PD_dataset/1 import data/outlier_plots_after")

###################
# Age plots
###################
x <- gse99039_pData[order(as.numeric(gse99039_pData$`age at exam:ch1`)),]
png('Sample_age_GSE99039.png', width = 958, height = 614)
plot(x$`age at exam:ch1`, col = factor(x$`disease label:ch1`), xlab = "", ylab = "age", main="Plot of ages of samples GSE99039 (197 are NA)", xaxt='n')
legend(1, 80, legend=c("CTL", "PD"), pch =1,  col=c("black", "red"))
dev.off()

###################
# Density Plots
###################
brewer.cols <- brewer.pal(9, "Set1")
png("Unprocessed log scale probe intensities.png", height = 1200, width = 1800)
par(mar=c(5.1,5.1,4.1,2.1))
boxplot(unfiltered.raw.data, col = brewer.cols, ylab = "Unprocessed log (base 2) scale Probe Intensities", xlab = "Array Names", main = "Unprocessed log (base 2) scale Probe Intensities of each Array", xaxt='n', cex.main=3.5, cex.lab=3, cex.axis=2)
dev.off()
#Contruct density plots
png("Unprocessed density plots.png", height = 1200, width = 1800)
par(mar=c(5.1,5.1,4.1,2.1))
hist(unfiltered.raw.data, main = "Density plot of log(2) probe intensities", col = brewer.cols, lty=1, xlab = "Log (base 2) Intensities", lwd =3,  cex.main=3.5, cex.lab=3, cex.axis=2)
dev.off()

###########################
#	Save data
###########################
setwd("C:/Users/Jack/Desktop/Deep_learning/PD_dataset/1 import data")
save(unfiltered.raw.data, file = "unfiltered.raw.data.Rdata")
write.table(gse99039_pData, file = "gse99039_pData.txt", append = FALSE, sep = "\t", dec = ".",
            row.names = TRUE, col.names = TRUE, quote = F)
save(gse99039_pData, file = "gse99039_pData.Rdata")
