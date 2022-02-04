#Steps here
##RMA normalised
##extract AD and HC patients
##extract Western European and Caucasian ethnicity patients
##The effect of the age and gender were controlled for using the ComBat function in the sva R package
##Probesets without annotation (Entrez_Gene_ID) were filtered out
##any genes with multiple probes keep highest MAD
#not removing bottom 5% due to this paper - https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007264#abstract0

#import packages
library(limma) 
library(sva)
library(illuminaHumanv3.db)

###############################
#  Import the data filtered
###############################
setwd("C:/Users/Jack/Desktop/Deep_learning/AD_ML/1 import data/outlier_plots_after")
load("data.RData")


###############################
#  Normalise the data
###############################
raw.data.log <- as.matrix(log2(GSE63060))
y <- limma::backgroundCorrect(raw.data.log ,method="normexp", offset = 0, normexp.method = "rma")
GSE63060 <- normalizeBetweenArrays(y, method ="quantile")

raw.data.log <- as.matrix(log2(GSE63061))
y <- limma::backgroundCorrect(raw.data.log ,method="normexp", offset = 0, normexp.method = "rma")
GSE63061 <- normalizeBetweenArrays(y, method ="quantile")


###############################
#  Correcting for gender
###############################
mod = model.matrix(~as.factor(`status`), data=gse63060_pData)
batch = gse63060_pData$`gender`
GSE63060 = ComBat(dat=GSE63060, batch=batch, mod=mod, par.prior=TRUE, prior.plots=FALSE)

mod = model.matrix(~as.factor(`status`), data=gse63061_pData)
batch = gse63061_pData$`gender`
GSE63061 = ComBat(dat=GSE63061, batch=batch, mod=mod, par.prior=TRUE, prior.plots=FALSE)


###############################
#  Extract AD, MCI and HC samples
###############################
###############
#GSE63060
###############
unique(gse63060_pData$ethnicity)
#Western European, Other Caucasian,  Unknown 
gse63060_pData <- rbind(gse63060_pData[gse63060_pData$ethnicity == "Western European" ,], gse63060_pData[gse63060_pData$ethnicity == "Other Caucasian" ,])
gse63060_pData <- rbind(gse63060_pData[gse63060_pData$status == "CTL" ,], gse63060_pData[gse63060_pData$status == "AD" ,])

GSE63060 <- GSE63060[,rownames(gse63060_pData)]

nrow(gse63060_pData) #247
nrow(gse63060_pData[gse63060_pData$status == "AD",]) #143
nrow(gse63060_pData[gse63060_pData$status == "CTL",]) #104

###############
#GSE63061
###############
unique(gse63061_pData$ethnicity)
#Western European, British_Scottish, British_English, Other Caucasian,  Unknown , Irish, Any_Other_White_Background, British_Other_Background, British_Welsh, Indian, British English, British, Caribbean , Any_Other_Ethnic_Background, Asian, White_And_Asian, Any_Other_Black_Background, Any_Other_Asian_Background, unkown but she's white and speaks english with a slight south african accent

gse63061_pData <- rbind(
gse63061_pData[gse63061_pData$ethnicity == "Western European" ,], 
gse63061_pData[gse63061_pData$ethnicity == "Other Caucasian" ,],
gse63061_pData[gse63061_pData$ethnicity == "British_Scottish" ,],
gse63061_pData[gse63061_pData$ethnicity == "British_English" ,],
gse63061_pData[gse63061_pData$ethnicity == "Irish" ,],
gse63061_pData[gse63061_pData$ethnicity == "Any_Other_White_Background" ,],
gse63061_pData[gse63061_pData$ethnicity == "British_Other_Background" ,],
gse63061_pData[gse63061_pData$ethnicity == "British_Welsh" ,],
gse63061_pData[gse63061_pData$ethnicity == "British English" ,],
gse63061_pData[gse63061_pData$ethnicity == "British" ,]
)

gse63061_pData <- rbind(gse63061_pData[gse63061_pData$status == "CTL" ,], gse63061_pData[gse63061_pData$status == "AD" ,])

GSE63061 <- GSE63061[,rownames(gse63061_pData)]

nrow(gse63061_pData) #268
nrow(gse63061_pData[gse63061_pData$status == "AD",]) #137
nrow(gse63061_pData[gse63061_pData$status == "CTL",]) #131



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


#GSE63060
probes=row.names(GSE63060)
Symbols = unlist(mget(probes, illuminaHumanv3SYMBOL, ifnotfound=NA))
Entrez_IDs = unlist(mget(probes, illuminaHumanv3ENTREZID, ifnotfound=NA))
sorted_data.annotate =cbind(Entrez_IDs,Symbols, GSE63060)

sorted_data_annotate = remove(sorted_data.annotate)
#29454 probes for 19233 unique genes
sorted_annotate = sorted_data_annotate[,3:ncol(sorted_data_annotate)]
rownames(sorted_annotate) = sorted_data_annotate[,2]
mode(sorted_annotate) <- "numeric" 
GSE63060 <- sorted_annotate


#GSE63061
probes=row.names(GSE63061)
Symbols = unlist(mget(probes, illuminaHumanv3SYMBOL, ifnotfound=NA))
Entrez_IDs = unlist(mget(probes, illuminaHumanv3ENTREZID, ifnotfound=NA))
sorted_data.annotate =cbind(Entrez_IDs,Symbols, GSE63061)

sorted_data_annotate = remove(sorted_data.annotate)
#29333 probes for 246 unique genes
sorted_annotate = sorted_data_annotate[,3:ncol(sorted_data_annotate)]
rownames(sorted_annotate) = sorted_data_annotate[,2]
mode(sorted_annotate) <- "numeric" 
GSE63061 <- sorted_annotate




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


#GSE63060
MAD <- vector(mode="numeric", length=0)
for( i in 1:29454){                
        MAD[i] <- mad(GSE63060[i,1:247])
}
ExprsMAD <- cbind(MAD, GSE63060)
ExprsMAD <- uniqGeneId(ExprsMAD)   #19233 probes
GSE63060 <- ExprsMAD[,2:ncol(ExprsMAD)]

#GSE63061
MAD <- vector(mode="numeric", length=0)
for( i in 1:29333){                
        MAD[i] <- mad(GSE63061[i,1:268])
}
ExprsMAD <- cbind(MAD, GSE63061)
ExprsMAD <- uniqGeneId(ExprsMAD)   #19146 probes
GSE63061 <- ExprsMAD[,2:ncol(ExprsMAD)]


###############################
#  Finalize and save datasets
###############################
setwd("C:/Users/Jack/Desktop/Deep_learning/AD_ML/2 prepared data/AD HC")

GSE63060 <- GSE63060[,rownames(gse63060_pData)]
GSE63060 <- t(GSE63060)
save(GSE63060, file = "GSE63060.Rdata")
save(gse63060_pData, file = "gse63060_pData.Rdata")

write.table(GSE63060, file="GSE63060.txt", row.names=T, col.names=T, sep = "\t")
write.table(gse63060_pData, file="gse63060_pData.txt", row.names=T, col.names=T, sep = "\t", quote = F)


GSE63061 <- GSE63061[,rownames(gse63061_pData)]
GSE63061 <- t(GSE63061)
save(GSE63061, file = "GSE63061.Rdata")
save(gse63061_pData, file = "gse63061_pData.Rdata")

write.table(GSE63061, file="GSE63061.txt", row.names=T, col.names=T, sep = "\t")
write.table(gse63061_pData, file="gse63061_pData.txt", row.names=T, col.names=T, sep = "\t", quote = F)
