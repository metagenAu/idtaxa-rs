# Read `idtaxa classify` output into a DECIPHER-compatible object so that
# downstream R code written for IdTaxa(type = "extended") works unchanged.
#
#   source("scripts/read_idtaxa_rs.R")
#   ids <- read_idtaxa_rs("out.tsv")             # class c("Taxa", "Test")
#   ids <- read_idtaxa_rs("out.tsv", "rank_names.txt")
#
# Native TSV omits Root and the "unclassified_*" filler; both are restored
# here with DECIPHER's conventions (Root's confidence is taken as the first
# rank's, which it equals unless several first-rank taxa compete). The
# output of `idtaxa classify --format decipher` (collapsed strings) can be
# read with read.delim(..., header = FALSE) directly, like
# IdTaxa(type = "collapsed").
read_idtaxa_rs <- function(path, ranks = NULL) {
  d <- read.delim(path, comment.char = "#", quote = "", colClasses = "character")
  full <- "full_taxonomy" %in% names(d)
  res <- lapply(seq_len(nrow(d)), function(i) {
    tax <- if (nzchar(d$taxonomy[i])) strsplit(d$taxonomy[i], ";", fixed = TRUE)[[1]] else character()
    conf <- if (nzchar(d$confidences[i])) as.numeric(strsplit(d$confidences[i], ";", fixed = TRUE)[[1]]) else numeric()
    root_conf <- if (length(conf)) 100 else 0
    if (full && nzchar(d$full_taxonomy[i])) {
      ft <- strsplit(d$full_taxonomy[i], ";", fixed = TRUE)[[1]]
      fc <- as.numeric(strsplit(d$full_confidences[i], ";", fixed = TRUE)[[1]])
      if (length(fc)) root_conf <- max(fc[1], conf[1], na.rm = TRUE)
    } else ft <- tax
    taxon <- c("Root", tax)
    confidence <- c(root_conf, conf)
    if (d$strand[i] == "NA") {
      taxon <- c("Root", "unclassified_Root"); confidence <- c(0, 0)
    } else if (length(ft) > length(tax)) {
      taxon <- c(taxon, paste0("unclassified_", taxon[length(taxon)]))
      confidence <- c(confidence, confidence[length(confidence)])
    }
    x <- list(taxon = taxon, confidence = confidence)
    if (!is.null(ranks)) x$rank <- ranks[seq_along(taxon)]
    x
  })
  names(res) <- d$id
  class(res) <- c("Taxa", "Test")
  res
}
