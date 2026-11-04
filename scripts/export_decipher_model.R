#!/usr/bin/env Rscript
# Export a DECIPHER Taxa/Train object (from LearnTaxa) to the neutral
# directory format read by `idtaxa import-decipher` (see
# crates/idtaxa-core/src/import.rs for the specification).
#
# Usage: Rscript export_decipher_model.R trainingSet.RData|.rds out_dir [object_name]
suppressPackageStartupMessages({ library(DECIPHER); library(jsonlite) })
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("usage: export_decipher_model.R model.(rds|RData) out_dir [object_name]")
inp <- args[1]; out <- args[2]
if (grepl("\\.rds$", inp, ignore.case = TRUE)) {
  ts <- readRDS(inp)
} else {
  env <- new.env(); nm <- load(inp, envir = env)
  obj <- if (length(args) >= 3) args[3] else nm[1]
  ts <- get(obj, envir = env)
}
if (!is(ts, "Taxa") || !is(ts, "Train")) stop("object is not a DECIPHER Taxa/Train object")
if (!is.null(ts$alphabet)) stop("amino-acid models are not supported")
dir.create(out, showWarnings = FALSE, recursive = TRUE)

wi <- function(x, f) { con <- file(file.path(out, f), "wb"); writeBin(as.integer(x), con, size = 4, endian = "little"); close(con) }
wd <- function(x, f) { con <- file(file.path(out, f), "wb"); writeBin(as.double(x), con, size = 8, endian = "little"); close(con) }

wi(lengths(ts$kmers), "kmer_lengths.bin")
wi(unlist(ts$kmers, use.names = FALSE), "kmers.bin")
wd(ts$IDFweights, "idf.bin")

nodes <- length(ts$taxonomy)
dl <- integer(nodes); dk <- vector("list", nodes); dw <- vector("list", nodes)
for (k in seq_len(nodes)) {
  d <- ts$decisionKmers[[k]]
  if (is.null(d) || length(d[[1]]) == 0) next
  km <- d[[1]]; m <- d[[2]]
  nc <- length(ts$children[[k]])
  if (!is.matrix(m)) m <- matrix(m, nrow = nc)
  # LearnTaxa's t(profile[keep, ]) drops dimensions when one k-mer is kept
  if (nrow(m) != nc && ncol(m) == nc && length(km) == 1) m <- t(m)
  if (nrow(m) != nc || ncol(m) != length(km)) stop("unexpected decision matrix shape at node ", k)
  dl[k] <- length(km); dk[[k]] <- km; dw[[k]] <- as.vector(t(m)) # row-major
}
wi(dl, "decision_lengths.bin")
wi(unlist(dk), "decision_kmers.bin")
wd(unlist(dw), "decision_weights.bin")

ps <- ts$problemSequences
meta <- list(
  format = unbox("idtaxa-decipher-export"),
  version = unbox(1L),
  decipher_version = unbox(as.character(packageVersion("DECIPHER"))),
  K = unbox(as.integer(ts$K)),
  taxonomy = ts$taxonomy,
  taxa = ts$taxa,
  ranks = if (is.null(ts$ranks)) NULL else ts$ranks,
  levels = as.integer(ts$levels),
  parents = as.integer(ts$parents),
  fraction = as.double(ts$fraction),
  crossIndex = as.integer(ts$crossIndex),
  children = lapply(ts$children, as.integer),
  problemSequences = list(Index = as.integer(ps$Index), Expected = as.character(ps$Expected),
                          Predicted = as.character(ps$Predicted)),
  problemGroups = as.character(ts$problemGroups),
  names = if (is.null(names(ts$kmers))) NULL else names(ts$kmers),
  alphabet = NULL
)
writeLines(toJSON(meta, na = "null", null = "null", digits = NA), file.path(out, "meta.json"))
cat("exported", length(ts$kmers), "training sequences,", nodes, "taxa, K =", ts$K, "to", out, "\n")
