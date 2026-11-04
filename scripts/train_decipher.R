#!/usr/bin/env Rscript
# Train a DECIPHER model with LearnTaxa (defaults) and save it as RDS.
# Usage: Rscript train_decipher.R train.fasta train_taxonomy.tsv out.rds [seed] [K]
suppressPackageStartupMessages(library(DECIPHER))
a <- commandArgs(trailingOnly = TRUE)
seqs <- readDNAStringSet(a[1])
tax <- read.delim(a[2], header = FALSE, sep = "\t", quote = "", colClasses = "character")
m <- match(names(seqs), tax[[1]])
if (anyNA(m)) stop("sequences without taxonomy")
set.seed(if (length(a) >= 4) as.integer(a[4]) else 1)
t0 <- Sys.time()
K <- if (length(a) >= 5) as.integer(a[5]) else NULL
ts <- LearnTaxa(seqs, tax[[2]][m], K = K, verbose = FALSE)
names(ts$kmers) <- names(seqs)
cat("LearnTaxa seconds:", as.numeric(difftime(Sys.time(), t0, units = "secs")), "\n")
print(ts)
saveRDS(ts, a[3])
