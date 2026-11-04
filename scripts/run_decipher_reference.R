#!/usr/bin/env Rscript
# Run DECIPHER IdTaxa on queries and write a TSV comparable to
# `idtaxa classify --full-confidences` (threshold 0 so every rank's
# confidence is reported; thresholds are applied afterwards by compare.py).
# Usage: Rscript run_decipher_reference.R model.rds queries.fasta out.tsv [seed] [processors] [threshold] [strand]
suppressPackageStartupMessages(library(DECIPHER))
a <- commandArgs(trailingOnly = TRUE)
ts <- readRDS(a[1])
q <- readDNAStringSet(a[2])
seed <- if (length(a) >= 4) as.integer(a[4]) else 1L
procs <- if (length(a) >= 5 && a[5] != "NULL") as.integer(a[5]) else NULL
thr <- if (length(a) >= 6) as.numeric(a[6]) else 0
strand <- if (length(a) >= 7) a[7] else "both"
set.seed(seed)
t0 <- Sys.time()
res <- IdTaxa(q, ts, strand = strand, threshold = thr, processors = procs, verbose = FALSE)
secs <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
cat("IdTaxa seconds:", secs, "for", length(q), "queries\n")
out <- data.frame(
  id = names(q),
  taxonomy = vapply(res, function(x) paste(x$taxon[-1], collapse = ";"), ""),
  confidences = vapply(res, function(x) paste(sprintf("%.4f", x$confidence[-1]), collapse = ";"), ""),
  stringsAsFactors = FALSE
)
write.table(out, a[3], sep = "\t", quote = FALSE, row.names = FALSE)
writeLines(as.character(secs), paste0(a[3], ".seconds"))
