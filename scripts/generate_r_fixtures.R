#!/usr/bin/env Rscript
# Generate reference fixtures for pyFIES parity tests.
#
# Runs RM.weights on the four bundled FAO sample countries and dumps the key
# numerical outputs to JSON files under tests/fixtures/r_reference/.
#
# Run once after installing R + RM.weights:
#
#     Rscript scripts/generate_r_fixtures.R
#
# Requires: RM.weights, jsonlite. Both are installed automatically below if
# missing.

suppressPackageStartupMessages({
  if (!requireNamespace("RM.weights", quietly = TRUE)) {
    install.packages("RM.weights", repos = "https://cloud.r-project.org")
  }
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    install.packages("jsonlite", repos = "https://cloud.r-project.org")
  }
  library(RM.weights)
  library(jsonlite)
})

set.seed(20260427)  # deterministic for the small handful of randomised paths

out_dir <- file.path("tests", "fixtures", "r_reference")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

dump_country <- function(country_idx) {
  data_name <- paste0("data.FAO_country", country_idx)
  data(list = data_name, package = "RM.weights")
  d <- get(data_name)
  XX <- as.matrix(d[, 1:8])
  wt <- d$wt

  rr <- RM.w(XX, wt)
  pp <- prob.assign(rr, sthres = c(rr$b[5], rr$b[8]))

  # Equate to the FAO 2014-2016 global standard (default `st`).
  ee <- equating.fun(rr)

  cat(sprintf("country %d: n=%d, n_complete=%d, converged=%s\n",
              country_idx, nrow(XX), rr$n.compl, rr$converged))

  # Pack the input data matrix and weights so pyFIES can re-fit and diff.
  # NA values become JSON null; pyFIES reloads them as NaN.
  out <- list(
    country = country_idx,
    n_total = nrow(XX),
    n_complete = rr$n.compl,
    items = colnames(XX),
    # Input data (for parity replay in pyFIES)
    data = unname(XX),  # row-major matrix of 0/1/NA
    weights = unname(wt),
    # CML fit
    beta = unname(rr$b),
    se_beta = unname(rr$se.b),
    theta = unname(rr$a),
    se_theta = unname(rr$se.a),
    pseudo_extreme = unname(rr$d),
    infit = unname(rr$infit),
    outfit = unname(rr$outfit),
    reliab = rr$reliab,
    reliab_fl = rr$reliab.fl,
    weighted_raw_score_dist = unname(rr$wt.rs),
    converged = rr$converged,
    # Prevalence at country thresholds
    sthres_country = c(rr$b[5], rr$b[8]),
    sprob_country = unname(pp$sprob),
    raw_score_dist = unname(pp$f_j),
    # Equating to global standard
    equate_scale = ee$scale,
    equate_shift = ee$shift,
    equate_common = unname(ee$common),
    equate_adj_thres = unname(ee$adj.thres),
    equate_prevs = unname(ee$prevs),
    equate_correlation = ee$cor.comm.items,
    standard = unname(ee$standard)
  )

  fp <- file.path(out_dir, sprintf("country%d.json", country_idx))
  write_json(out, fp, pretty = TRUE, digits = NA, auto_unbox = TRUE)
  cat(sprintf("  -> %s\n", fp))
}

for (i in 1:4) {
  dump_country(i)
}

cat("\nFixtures written to:", out_dir, "\n")
cat("Commit them under tests/fixtures/r_reference/ for CI parity checks.\n")
