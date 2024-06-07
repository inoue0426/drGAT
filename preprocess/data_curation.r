# Execute this script using Rscript data_curation.r to extract drug response, mutation, log2 gene expression, and copy data, and then run preprocess.py.
# To run this script, open the terminal and type "Rscript data_curation.r".

library(rcellminer) # Load the rcellminer library
library(rcellminerData) # Load the rcellminerData library

drugAct <- exprs(getAct(rcellminerData::drugData)) # Get drug activity data
write.csv(drugAct, 'nci60Act.csv') # Save as a CSV file

convert_log2_to_fpkm <- function(log2_values) {
    result <- 2^log2_values - 1
    return(result)
}

expression <- getAllFeatureData(rcellminerData::molData)[["xsq"]] # Retrieve log2 gene expression data
expression <- round(convert_log2_to_fpkm(expression), 2)
names <- read.csv("simplified_names.csv", stringsAsFactors = FALSE)
colnames(expression) <- sapply(colnames(expression), function(x) names[names[,1] == x, 2])
expression <- expression[order(row.names(expression)),]

write.csv(expression, 'nci60_gene_exp.csv') # Save as a CSV file

