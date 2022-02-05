#import packages
library(GEOquery)
library(stringr)
library(ggplot2)
library(ggpubr)

#import the phenodata from R
setwd("E:/PhD work/AD ML/import data")


################################################
# 					sort out the phenodata 						 #
################################################


gse63060_pData <- getGEO("GSE63060", GSEMatrix = TRUE)
gse63060_pData <- pData(phenoData(gse63060_pData[[1]]))[,c(1,35, 36, 37, 39)]
rownames(gse63060_pData) <- gse63060_pData$title
gse63060_pData <- gse63060_pData[,2:5]
colnames(gse63060_pData) <- c("age", "ethnicity", "gender", "status")

HCpData <- gse63060_pData[gse63060_pData$status  == "CTL",]
ADpData <- gse63060_pData[gse63060_pData$status  == "AD",]
MCIpData <- gse63060_pData[gse63060_pData$status  == "MCI",]
gse63060_pData <- rbind(ADpData, MCIpData, HCpData)


gse63061_pData <- getGEO("GSE63061", GSEMatrix = TRUE)
gse63061_pData <- pData(phenoData(gse63061_pData[[1]]))[,c(1,35, 36, 37, 39)]
rownames(gse63061_pData) <- gse63061_pData$title
gse63061_pData <- gse63061_pData[,2:5]
colnames(gse63061_pData) <- c("age", "ethnicity", "gender", "status")

HCpData <- gse63061_pData[gse63061_pData$status  == "CTL",]
ADpData <- gse63061_pData[gse63061_pData$status  == "AD",]
MCIpData <- gse63061_pData[gse63061_pData$status  == "MCI",]
gse63061_pData <- rbind(ADpData, MCIpData, HCpData)

##when I import the data in the next step, as the col names begin with a number so are imported with 'X' in front, so add X to the sample names in pData
rownames(gse63060_pData) <- paste0("X", rownames(gse63060_pData), sep="")
rownames(gse63061_pData) <- paste0("X", rownames(gse63061_pData), sep="")

write.table(gse63060_pData, file = "gse63060_pData.txt", append = FALSE, sep = "\t", dec = ".",
            row.names = TRUE, col.names = TRUE, quote = F)
write.table(gse63061_pData, file = "gse63061_pData.txt", append = FALSE, sep = "\t", dec = ".",
            row.names = TRUE, col.names = TRUE, quote = F)




################################
#                              #
#         Import data          #
#                              #
################################
x <- read.table('GSE63060_non-normalized.txt', header = T, sep = "\t")
rownames(x) <- x[,1]
x <- x[,2:ncol(x)]
colnames(x) <- str_replace(colnames(x), ".AVG_Signal", "" )

# 3 columns are unlabelled. So need to allocate label to them
#import the normalized which is correctly labelled
y <- read.table('GSE63060_normalized.txt', header = T, sep = "\t")
rownames(y) <- y[,1]
y <- y[,2:ncol(y)]
#see which columns aren't shared by names
setdiff(colnames(x), colnames(y))
setdiff(colnames(y), colnames(x))
#see gene expression for the 3 columns that aren't labelled correctly
x[rownames(x)[4:6],c("X","X.1","X.2")]
y[rownames(x)[4:6],c("X4856050008_B", "X4856050008_K", "X4856050048_A")]
#X4856050008_B expression goes 7.3 -> 7.6 -> 7.5. This is similar to X
#X4856050008_K expression goes 7.35 -> 7.39 -> 7.51. This is similar to X.1
#X4856050048_A expression goes 7.45 -> 7.47 -> 7.44. This is simliar to X.2
##rename the columns in x
colnames(x)[colnames(x) == "X"] <- "X4856050008_B"
colnames(x)[colnames(x) == "X.1"] <- "X4856050008_K"
colnames(x)[colnames(x) == "X.2"] <- "X4856050048_A"
GSE63060 <- x

x <- read.table('GSE63061_non-normalized.txt', header = T, sep = "\t")
rownames(x) <- x[,1]
x <- x[,2:ncol(x)]
GSE63061 <- x


################################
#                              #
#   Plot to detect outliers    #
#                              #
################################
setwd("C:/Users/Jack/Desktop/Deep_learning/AD_dataset/1 import data/outlier_plots_before")

###################
#	Age plots
###################
x <- gse63060_pData[order(as.numeric(gse63060_pData$age)),]
png('Sample_age_GSE63060.png', width = 958, height = 614)
plot(x$age, col = factor(x$status), xlab = "", ylab = "age", main="Plot of ages of samples GSE63060", xaxt='n')
legend(1, 90, legend=c("CTL", "MCI", "AD"), pch =1,  col=c("red", "green", "black"))
dev.off()

x <- gse63061_pData[order(as.numeric(gse63061_pData$age)),]
png('Sample_age_GSE63061.png', width = 958, height = 614)
plot(x$age, col = factor(x$status), xlab = "", ylab = "age", main="Plot of ages of samples GSE63061", xaxt='n')
legend(1, 100, legend=c("CTL", "MCI", "AD"), pch =1,  col=c("red", "green", "black"))
dev.off()

###################
#	PCA plots
###################

######
#GSE63060
######

dataset <- GSE63060[,rownames(gse63060_pData)] #put in same order
exp_raw <- log2(dataset)
PCA_raw <- prcomp(t(exp_raw), scale. = FALSE)

percentVar <- round(100*PCA_raw$sdev^2/sum(PCA_raw$sdev^2),1)
sd_ratio <- sqrt(percentVar[2] / percentVar[1])

dataGG <- data.frame(PC1 = PCA_raw$x[,1], PC2 = PCA_raw$x[,2],
                    Disease = gse63060_pData$status,
                    Individual = rownames(gse63060_pData),
                    Gender = gse63060_pData$gender)

a <- ggplot(dataGG, aes(PC1, PC2)) +
    geom_point(aes(shape = Disease, colour = Gender)) +
  	xlab(paste0("PC1, VarExp: ", percentVar[1], "%")) +
  	ylab(paste0("PC2, VarExp: ", percentVar[2], "%")) +
  	theme(plot.title = element_text(hjust = 0.5))+
  	coord_fixed(ratio = sd_ratio) +
  	scale_color_manual(values = c("darkorange2", "dodgerblue4"))

png('PCA_plot_GSE63060.png', width = 854, height = 628)
ggplot(dataGG, aes(PC1, PC2)) +
    geom_point(aes(shape = Disease, colour = Gender)) +
  	xlab(paste0("PC1, VarExp: ", percentVar[1], "%")) +
  	ylab(paste0("PC2, VarExp: ", percentVar[2], "%")) +
  	labs(title="PCA plot of the log-transformed GSE63060 raw expression data") +
  	theme(plot.title = element_text(size = 20, hjust = 0.5))+
  	scale_color_manual(values = c("darkorange2", "dodgerblue4"))
dev.off()

######
#GSE63061
######

dataset <- GSE63061[,rownames(gse63061_pData)] #put in same order
exp_raw1 <- log2(dataset)
PCA_raw1 <- prcomp(t(exp_raw1), scale. = FALSE)

percentVar1 <- round(100*PCA_raw1$sdev^2/sum(PCA_raw1$sdev^2),1)
sd_ratio1 <- sqrt(percentVar1[2] / percentVar1[1])

dataGG1 <- data.frame(PC1 = PCA_raw1$x[,1], PC2 = PCA_raw1$x[,2],
                    Disease = gse63061_pData$status,
                    Individual = rownames(gse63061_pData),
                    Gender = gse63061_pData$gender)

b <- ggplot(dataGG1, aes(PC1, PC2)) +
    geom_point(aes(shape = Disease, colour = Gender)) +
  	xlab(paste0("PC1, VarExp: ", percentVar1[1], "%")) +
  	ylab(paste0("PC2, VarExp: ", percentVar1[2], "%")) +
  	theme(plot.title = element_text(hjust = 0.5))+
  	coord_fixed(ratio = sd_ratio1) +
  	scale_color_manual(values = c("darkorange2", "dodgerblue4"))

png('PCA_plot_GSE63061.png', width = 854, height = 628)
ggplot(dataGG1, aes(PC1, PC2)) +
    geom_point(aes(shape = Disease, colour = Gender)) +
  	xlab(paste0("PC1, VarExp: ", percentVar1[1], "%")) +
  	ylab(paste0("PC2, VarExp: ", percentVar1[2], "%")) +
  	labs(title="PCA plot of the log-transformed GSE63061 raw expression data") +
  	theme(plot.title = element_text(size = 20, hjust = 0.5))+
  	scale_color_manual(values = c("darkorange2", "dodgerblue4"))
dev.off()


#######
#plotting
#######
png('PCA_plot_sep.png', width = 958, height = 614)
figure <- ggarrange(a, b, 
          labels = c("GSE63060", "GSE63061"),
          ncol = 1, nrow = 2)
annotate_figure(figure, top = text_grob("PCA plot of the log-transformed raw expression data", color = "black", face = "bold", size = 20))
dev.off()



###################
#	Density Plots
###################
density_plots <- function(x, height) {
	#create a data frame of two columns, one with expression and one with samples ID
	y <- data.frame(x[,1])
	y$data <- colnames(x)[1]
	colnames(y)[1] <- "col"
	datas <- y
	#itteratively add each sample onto this data.frame
	for (p in 2:ncol(x)){
		l <- data.frame(x[,p])
		l$data <- colnames(x)[p]
		colnames(l)[1] <- "col"
		datas <- rbind(datas, l)
	}
	#log the expression values
	datas$col <- log2(datas$col)
	#density plot of the data
	ggplot(datas, aes(col, fill = data)) +
		geom_density(alpha = 0.2) +
		theme(legend.position="none") +
		scale_y_continuous(expand = c(0, 0), limits = c(0, height)) +
		scale_x_continuous(limits = c(5.8, 11)) +
		labs(y="Density", x = "Log2 intensity")
}

###
#GSE63060
###
png('GSE63060_density.png', width = 1269, height = 667)
density_plots(GSE63060, height = 3.5)
dev.off()

AD <- density_plots(GSE63060[,rownames(gse63060_pData[gse63060_pData$status == "AD",])], height = 3.5)
MCI <- density_plots(GSE63060[,rownames(gse63060_pData[gse63060_pData$status == "MCI",])], height = 3.5)
CTL <- density_plots(GSE63060[,rownames(gse63060_pData[gse63060_pData$status == "CTL",])], height = 3.5)

png('GSE63060_density_sep.png', width = 1269, height = 667)
figure <- ggarrange(AD, MCI, CTL, density_plots(GSE63060, height = 3.5), 
          labels = c("AD", "MCI", "CTL", "All"),
          ncol = 2, nrow = 2)
annotate_figure(figure, top = text_grob("GSE63060", color = "black", face = "bold", size = 20))
dev.off()


###
#GSE63061
###
png('GSE63061_density.png', width = 1269, height = 667)
density_plots(GSE63061, height = 6.3)
dev.off()

AD <- density_plots(GSE63061[,rownames(gse63061_pData[gse63061_pData$status == "AD",])], height = 6.3)
MCI <- density_plots(GSE63061[,rownames(gse63061_pData[gse63061_pData$status == "MCI",])], height = 6.3)
CTL <- density_plots(GSE63061[,rownames(gse63061_pData[gse63061_pData$status == "CTL",])], height = 6.3)

png('GSE63061_density_sep.png', width = 1269, height = 667)
figure <- ggarrange(AD, MCI, CTL, density_plots(GSE63061, height = 6.3),
          labels = c("AD", "MCI", "CTL", "All"),
          ncol = 2, nrow = 2)
annotate_figure(figure, top = text_grob("GSE63061", color = "black", face = "bold", size = 20))
dev.off()


##########
# Remove the AD sample outlier
##########
temp <- GSE63061[,rownames(gse63061_pData[gse63061_pData$status == "AD",])]
temp <- log2(temp)
ggplot(temp, aes(temp[,4], fill = "red")) +
		geom_density(alpha = 0.2) +
		theme(legend.position="none") +
		scale_y_continuous(expand = c(0, 0), limits = c(0, 6.3)) +
		scale_x_continuous(limits = c(5.8, 11)) +
		labs(y="Density", x = "Log2 intensity")
colnames(temp)[4] #X7943280031_H
##########
# Remove the MCI sample outlier
##########
temp <- GSE63061[,rownames(gse63061_pData[gse63061_pData$status == "MCI",])]
temp <- log2(temp)
ggplot(temp, aes(temp[,18], fill = "red")) +
		geom_density(alpha = 0.2) +
		theme(legend.position="none") +
		scale_y_continuous(expand = c(0, 0), limits = c(0, 6.3)) +
		scale_x_continuous(limits = c(5.8, 11)) +
		labs(y="Density", x = "Log2 intensity")
colnames(temp)[18] #X7196843054_A
ggplot(temp, aes(temp[,50], fill = "red")) +
		geom_density(alpha = 0.2) +
		theme(legend.position="none") +
		scale_y_continuous(expand = c(0, 0), limits = c(0, 6.3)) +
		scale_x_continuous(limits = c(5.8, 11)) +
		labs(y="Density", x = "Log2 intensity")
colnames(temp)[50] #X7348931018_J


###################
#	Box Plots
###################
box_plots <- function(x, bottom, height) {
	#create a data frame of two columns, one with expression and one with samples ID
	y <- data.frame(x[,1])
	y$data <- colnames(x)[1]
	colnames(y)[1] <- "col"
	datas <- y
	#itteratively add each sample onto this data.frame
	for (p in 2:ncol(x)){
		l <- data.frame(x[,p])
		l$data <- colnames(x)[p]
		colnames(l)[1] <- "col"
		datas <- rbind(datas, l)
	}
	#log the expression values
	datas$col <- log2(datas$col)
	#box plot of the data

	ggplot(datas, aes(x=data, y=col)) +
		geom_boxplot() +
		theme(legend.position="none") +
		labs(y="Log2 intensity", x = "Sample") +
		scale_y_continuous(expand = c(0, 0), limits = c(bottom, height)) +
		theme(axis.text.x=element_blank(), axis.ticks.x=element_blank())
}

###
#GSE63061
###
AD <- box_plots(GSE63061[,rownames(gse63061_pData[gse63061_pData$status == "AD",])], bottom = 5, height = 15)
MCI <- box_plots(GSE63061[,rownames(gse63061_pData[gse63061_pData$status == "MCI",])], bottom = 5, height = 15)
CTL <- box_plots(GSE63061[,rownames(gse63061_pData[gse63061_pData$status == "CTL",])], bottom = 5, height = 15)

png('GSE63061_boxplots_sep.png', width = 1269, height = 667)
figure <- ggarrange(AD, MCI, CTL,
          labels = c("AD", "MCI", "CTL"),
          ncol = 2, nrow = 2)
annotate_figure(figure, top = text_grob("GSE63061", color = "black", face = "bold", size = 20))
dev.off()

####
#GSE63060
####
AD <- box_plots(GSE63060[,rownames(gse63060_pData[gse63060_pData$status == "AD",])], bottom = 5.7, height = 16)
MCI <- box_plots(GSE63060[,rownames(gse63060_pData[gse63060_pData$status == "MCI",])], bottom = 5.7, height = 16)
CTL <- box_plots(GSE63060[,rownames(gse63060_pData[gse63060_pData$status == "CTL",])], bottom = 5.7, height = 16)

png('GSE63060_boxplots_sep.png', width = 1269, height = 667)
figure <- ggarrange(AD, MCI, CTL,
          labels = c("AD", "MCI", "CTL"),
          ncol = 2, nrow = 2)
annotate_figure(figure, top = text_grob("GSE63060", color = "black", face = "bold", size = 20))
dev.off()


###################
#	MA Plots
###################
#so much data, MA plot not usefull



################################
#                              #
#   Remoive outliers           #
#                              #
################################
###
#GSE63060
###
GSE63060 <- GSE63060[,rownames(gse63060_pData)]

###
#GSE63061
###
#remove 1 AD and 2 MCI samples
rem <- c("X7196843054_A", "X7348931018_J", "X7943280031_H")
row_nam <- rownames(gse63061_pData)[!rownames(gse63061_pData) %in% rem]
gse63061_pData <- gse63061_pData[row_nam,]
GSE63061 <- GSE63061[,rownames(gse63061_pData)]



################################
#                              #
#   Plot with outliers removed #
#                              #
################################
setwd("C:/Users/Jack/Desktop/Deep_learning/AD_dataset/1 import data/outlier_plots_after")

###################
#	Age plots
###################
x <- gse63060_pData[order(as.numeric(gse63060_pData$age)),]
png('Sample_age_GSE63060.png', width = 958, height = 614)
plot(x$age, col = factor(x$status), xlab = "", ylab = "age", main="Plot of ages of samples GSE63060", xaxt='n')
legend(1, 90, legend=c("CTL", "MCI", "AD"), pch =1,  col=c("red", "green", "black"))
dev.off()

x <- gse63061_pData[order(as.numeric(gse63061_pData$age)),]
png('Sample_age_GSE63061.png', width = 958, height = 614)
plot(x$age, col = factor(x$status), xlab = "", ylab = "age", main="Plot of ages of samples GSE63061", xaxt='n')
legend(1, 100, legend=c("CTL", "MCI", "AD"), pch =1,  col=c("red", "green", "black"))
dev.off()

###################
#	PCA plots
###################
######
#GSE63060
######

dataset <- GSE63060[,rownames(gse63060_pData)] #put in same order
exp_raw <- log2(dataset)
PCA_raw <- prcomp(t(exp_raw), scale. = FALSE)

percentVar <- round(100*PCA_raw$sdev^2/sum(PCA_raw$sdev^2),1)
sd_ratio <- sqrt(percentVar[2] / percentVar[1])

dataGG <- data.frame(PC1 = PCA_raw$x[,1], PC2 = PCA_raw$x[,2],
                    Disease = gse63060_pData$status,
                    Individual = rownames(gse63060_pData),
                    Gender = gse63060_pData$gender)

a <- ggplot(dataGG, aes(PC1, PC2)) +
    geom_point(aes(shape = Disease, colour = Gender)) +
  	xlab(paste0("PC1, VarExp: ", percentVar[1], "%")) +
  	ylab(paste0("PC2, VarExp: ", percentVar[2], "%")) +
  	theme(plot.title = element_text(hjust = 0.5))+
  	coord_fixed(ratio = sd_ratio) +
  	scale_color_manual(values = c("darkorange2", "dodgerblue4"))

png('PCA_plot_GSE63060.png', width = 854, height = 628)
ggplot(dataGG, aes(PC1, PC2)) +
    geom_point(aes(shape = Disease, colour = Gender)) +
  	xlab(paste0("PC1, VarExp: ", percentVar[1], "%")) +
  	ylab(paste0("PC2, VarExp: ", percentVar[2], "%")) +
  	labs(title="PCA plot of the log-transformed GSE63060 raw expression data") +
  	theme(plot.title = element_text(size = 20, hjust = 0.5))+
  	scale_color_manual(values = c("darkorange2", "dodgerblue4"))
dev.off()

######
#GSE63061
######
dataset <- GSE63061[,rownames(gse63061_pData)] #put in same order
exp_raw1 <- log2(dataset)
PCA_raw1 <- prcomp(t(exp_raw1), scale. = FALSE)

percentVar1 <- round(100*PCA_raw1$sdev^2/sum(PCA_raw1$sdev^2),1)
sd_ratio1 <- sqrt(percentVar1[2] / percentVar1[1])

dataGG1 <- data.frame(PC1 = PCA_raw1$x[,1], PC2 = PCA_raw1$x[,2],
                    Disease = gse63061_pData$status,
                    Individual = rownames(gse63061_pData),
                    Gender = gse63061_pData$gender)

b <- ggplot(dataGG1, aes(PC1, PC2)) +
    geom_point(aes(shape = Disease, colour = Gender)) +
  	xlab(paste0("PC1, VarExp: ", percentVar1[1], "%")) +
  	ylab(paste0("PC2, VarExp: ", percentVar1[2], "%")) +
  	theme(plot.title = element_text(hjust = 0.5))+
  	coord_fixed(ratio = sd_ratio1) +
  	scale_color_manual(values = c("darkorange2", "dodgerblue4"))

png('PCA_plot_GSE63061.png', width = 854, height = 628)
ggplot(dataGG1, aes(PC1, PC2)) +
    geom_point(aes(shape = Disease, colour = Gender)) +
  	xlab(paste0("PC1, VarExp: ", percentVar1[1], "%")) +
  	ylab(paste0("PC2, VarExp: ", percentVar1[2], "%")) +
  	labs(title="PCA plot of the log-transformed GSE63061 raw expression data") +
  	theme(plot.title = element_text(size = 20, hjust = 0.5))+
  	scale_color_manual(values = c("darkorange2", "dodgerblue4"))
dev.off()

#######
#plotting
#######
png('PCA_plot_sep.png', width = 958, height = 614)
figure <- ggarrange(a, b, 
          labels = c("GSE63060", "GSE63061"),
          ncol = 1, nrow = 2)
annotate_figure(figure, top = text_grob("PCA plot of the log-transformed raw expression data", color = "black", face = "bold", size = 20))
dev.off()



###################
#	Density Plots
###################
###
#GSE63060
###
png('GSE63060_density.png', width = 1269, height = 667)
density_plots(GSE63060, height = 3.5)
dev.off()

AD <- density_plots(GSE63060[,rownames(gse63060_pData[gse63060_pData$status == "AD",])], height = 3.5)
MCI <- density_plots(GSE63060[,rownames(gse63060_pData[gse63060_pData$status == "MCI",])], height = 3.5)
CTL <- density_plots(GSE63060[,rownames(gse63060_pData[gse63060_pData$status == "CTL",])], height = 3.5)

png('GSE63060_density_sep.png', width = 1269, height = 667)
figure <- ggarrange(AD, MCI, CTL, density_plots(GSE63060, height = 3.5), 
          labels = c("AD", "MCI", "CTL", "All"),
          ncol = 2, nrow = 2)
annotate_figure(figure, top = text_grob("GSE63060", color = "black", face = "bold", size = 20))
dev.off()

###
#GSE63061
###
png('GSE63061_density.png', width = 1269, height = 667)
density_plots(GSE63061, height = 6.3)
dev.off()

AD <- density_plots(GSE63061[,rownames(gse63061_pData[gse63061_pData$status == "AD",])], height = 6.3)
MCI <- density_plots(GSE63061[,rownames(gse63061_pData[gse63061_pData$status == "MCI",])], height = 6.3)
CTL <- density_plots(GSE63061[,rownames(gse63061_pData[gse63061_pData$status == "CTL",])], height = 6.3)

png('GSE63061_density_sep.png', width = 1269, height = 667)
figure <- ggarrange(AD, MCI, CTL, density_plots(GSE63061, height = 6.3),
          labels = c("AD", "MCI", "CTL", "All"),
          ncol = 2, nrow = 2)
annotate_figure(figure, top = text_grob("GSE63061", color = "black", face = "bold", size = 20))
dev.off()



###################
#	Box Plots
###################
###
#GSE63061
###
AD <- box_plots(GSE63061[,rownames(gse63061_pData[gse63061_pData$status == "AD",])], bottom = 5, height = 15)
MCI <- box_plots(GSE63061[,rownames(gse63061_pData[gse63061_pData$status == "MCI",])], bottom = 5, height = 15)
CTL <- box_plots(GSE63061[,rownames(gse63061_pData[gse63061_pData$status == "CTL",])], bottom = 5, height = 15)

png('GSE63061_boxplots_sep.png', width = 1269, height = 667)
figure <- ggarrange(AD, MCI, CTL,
          labels = c("AD", "MCI", "CTL"),
          ncol = 2, nrow = 2)
annotate_figure(figure, top = text_grob("GSE63061", color = "black", face = "bold", size = 20))
dev.off()

####
#GSE63060
####
AD <- box_plots(GSE63060[,rownames(gse63060_pData[gse63060_pData$status == "AD",])], bottom = 5.7, height = 16)
MCI <- box_plots(GSE63060[,rownames(gse63060_pData[gse63060_pData$status == "MCI",])], bottom = 5.7, height = 16)
CTL <- box_plots(GSE63060[,rownames(gse63060_pData[gse63060_pData$status == "CTL",])], bottom = 5.7, height = 16)

png('GSE63060_boxplots_sep.png', width = 1269, height = 667)
figure <- ggarrange(AD, MCI, CTL,
          labels = c("AD", "MCI", "CTL"),
          ncol = 2, nrow = 2)
annotate_figure(figure, top = text_grob("GSE63060", color = "black", face = "bold", size = 20))
dev.off()

###########################
#	Save data
###########################

save(gse63061_pData, gse63060_pData, GSE63060, GSE63061, file = "data.RData")
