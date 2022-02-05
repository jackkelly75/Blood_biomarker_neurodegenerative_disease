#Steps here
##RMA normalised
##extract PD and HC patients
## assign unknown genders using massiR
##The effect of the age and gender were controlled for using the ComBat function in the sva R package
##Probesets without annotation (Entrez_Gene_ID) were filtered out
##any genes with multiple probes keep highest MAD
#not removing bottom 5% due to this paper - https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007264#abstract0

#import packages
library(affy) 
library(RColorBrewer)
library(massiR)
library(hgu133plus2.db)
library(sva)

###############################
#  Import the data filtered
###############################
setwd("C:/Users/Jack/Desktop/Deep_learning/PD_ML/1_import_data")
load("unfiltered.raw.data.RData")
load("gse99039_pData.Rdata")

##############################
#  Normalise the data
##############################
data.rma.norm = rma(unfiltered.raw.data)
GSE99039 = exprs(data.rma.norm)

########################
#plot normalised data  #
########################
setwd("C:/Users/Jack/Desktop/Deep_learning/PD_ML_sex/2_prepared_data")
brewer.cols <- brewer.pal(9, "Set1")
png("RMA normalised log scale probe intensities.png", width = 1800, height = 1200)
par(mar=c(5.1,5.1,4.1,2.1))
boxplot(GSE99039, col = brewer.cols, ylab = "RMA normalised log (base 2) scale probe intensities", xlab = "Array names", main = "RMA normalised log (base 2) scale probe intensities of each array", xaxt='n', cex.main=3.5, cex.lab=3, cex.axis=2)
dev.off()
nrow(GSE99039) #54675

#########################
# Number of samples and cases
########################
nrow(gse99039_pData) #434
nrow(gse99039_pData[gse99039_pData$`disease label:ch1` == "IPD",]) #204
nrow(gse99039_pData[gse99039_pData$`disease label:ch1` == "CONTROL",]) #230


##########################
# Assign sex using massiR
##########################
hgu133Plus2.probes <- data.frame(y.probes["affy_hg_u133_plus_2"])
# use the massi.y function to calculate probe variation
massi_y_out <- massi_y(expression_data= data.rma.norm, y_probes=hgu133Plus2.probes)
# plot probe variation to aid in deciding on the most informative subset of y chromosome probes
massi_y_plot(massi_y_out)
# Extract the informative probes for clustering
massi_select_out <- massi_select(data.rma.norm, hgu133Plus2.probes, threshold=4)
# cluster samples to predict the sex for each sample
massi_cluster_out <- massi_cluster(massi_select_out)
# get the predicted sex for each sample
sex_prediction <- data.frame(massi_cluster_out[[2]])
rownames(sex_prediction) <- sex_prediction[,1]
sex_prediction <- sex_prediction[,2:ncol(sex_prediction)]

#testing the results 
trial <- merge(gse99039_pData, sex_prediction, by = "row.names")
sextest <- trial[,c("sex", "Sex:ch1")]
sextest$sex <- gsub("female", "F", sextest$sex)
sextest$sex <- gsub("male", "M", sextest$sex)
sextest$`Sex:ch1` <- gsub("Male", "M", sextest$`Sex:ch1`)
sextest$`Sex:ch1` <- gsub("Female", "F", sextest$`Sex:ch1`)
sextest <- na.omit(sextest)
sextest$hey <- NA
for (i in 1:nrow(sextest)){
  sextest$hey[i] <- paste(sextest$`Sex:ch1`[i], sextest$sex[i], sep = "")
}
sextest <- sextest[sextest$hey != "MM",]
sextest <- sextest[sextest$hey != "FF",]
#sextest now contains the number of mismatching genders
#14 different between them, an error of 3.5%  ((14/399)*100)
#35 samples don't have gender

#re creating pData with gender
gse99039_pData <-merge(gse99039_pData, sex_prediction, by = "row.names")
rownames(gse99039_pData) <- gse99039_pData[,1]
gse99039_pData <- gse99039_pData[,2:ncol(gse99039_pData)]
gse99039_pData$sex <- gsub("male", "Male", gse99039_pData$sex)
gse99039_pData$sex <- gsub("feMale", "Female", gse99039_pData$sex)
x <- gse99039_pData$`Sex:ch1`
y <- gse99039_pData$sex
for (i in 1:length(y)){
  if(is.na(x[i])){
    x[i] <- y[i]
  }
}
gse99039_pData$final_sex <- x


###############################
#  Correcting for gender
###############################
mod = model.matrix(~as.factor(`disease label:ch1`), data=gse99039_pData)
batch = gse99039_pData$`batch:ch1`
GSE99039 = ComBat(dat=GSE99039, batch=batch, mod=mod, par.prior=TRUE, prior.plots=FALSE)
batch = gse99039_pData$final_sex
GSE99039 = ComBat(dat=GSE99039, batch=batch, mod=mod, par.prior=TRUE, prior.plots=FALSE)

###############################
#  annotate the data and remove any NA
###############################
remove = function(x) {
	comma = grep(",", rownames(x))    #find any row with comma in them
        if(length(comma)>0){
        	x = x[-comma,]
        }
	space = grep(" ", rownames(x))    #find any row with space in them
        if(length(space)>0){
        	x = x[-space,]
        }
	semicolon = grep(";", rownames(x))    #find any row with semicolon in them
        if(length(semicolon)>0){
        	x = x[-semicolon,]
	}
	colon = grep(":", rownames(x))    #find any row with colon in them
        if(length(colon)>0){
        	x = x[-colon,]
        }
	i <- is.na(x[,1])
	x <- x[!i,]      # remove any probes that don't have a mapped gene	  
}

probes=row.names(GSE99039)     #get a list of probes
Symbols = unlist(mget(probes, hgu133plus2SYMBOL, ifnotfound=NA))   #get a list of symbols that match the probes of raw.data
Entrez_IDs = unlist(mget(probes, hgu133plus2ENTREZID, ifnotfound=NA))   #get a list of entrez IDs that match the probes of raw.data
sorted_data.annotate =cbind(Entrez_IDs,Symbols,GSE99039) 

sorted_data_annotate = remove(sorted_data.annotate)
#41932 probes for 20183 unique genes
sorted_annotate = sorted_data_annotate[,3:ncol(sorted_data_annotate)]
rownames(sorted_annotate) = sorted_data_annotate[,2]
mode(sorted_annotate) <- "numeric" 
GSE99039 <- sorted_annotate


###############################
#  annotate the data and remove any NA
###############################
uniqGeneId <- function(x) {
   x = x[order(rownames(x), abs(x[,2]), decreasing = TRUE), ]
   entrezID = unique(rownames(x))
   id = match(entrezID, rownames(x))
   x = x[id[!is.na(id)], ]
   x
}

MAD <- vector(mode="numeric", length=0)
for( i in 1:41932){                
        MAD[i] <- mad(GSE99039[i,1:434])
}
ExprsMAD <- cbind(MAD, GSE99039)
ExprsMAD <- uniqGeneId(ExprsMAD)   #20183 probes
GSE99039 <- ExprsMAD[,2:ncol(ExprsMAD)]



###############################
#  Finalize and save datasets
###############################
setwd("C:/Users/Jack/Desktop/Deep_learning/PD_ML_sex/2_prepared_data")
GSE99039 <- GSE99039[,rownames(gse99039_pData)]
GSE99039 <- t(GSE99039)
save(GSE99039, file = "GSE99039.Rdata")
save(gse99039_pData, file = "gse99039_pData.Rdata")

write.table(GSE99039, file="GSE99039.txt", row.names=T, col.names=T, sep = "\t")
write.table(gse99039_pData, file = "gse99039_pData.txt", append = FALSE, sep = "\t", dec = ".",
            row.names = TRUE, col.names = TRUE, quote = F)
