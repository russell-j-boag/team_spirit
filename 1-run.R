get_this_file <- function() {
  frame_files <- Filter(
    Negate(is.null),
    lapply(sys.frames(), function(frame) frame$ofile)
  )
  if (length(frame_files) > 0) {
    return(normalizePath(frame_files[[length(frame_files)]], winslash = "/", mustWork = TRUE))
  }

  args <- commandArgs(trailingOnly = FALSE)

  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    return(normalizePath(sub("^--file=", "", file_arg[[1]]), winslash = "/", mustWork = TRUE))
  }

  file_flags <- which(args %in% c("-f", "--file"))
  if (length(file_flags) > 0) {
    idx <- file_flags[[length(file_flags)]]
    if ((idx + 1) <= length(args)) {
      return(normalizePath(args[[idx + 1]], winslash = "/", mustWork = TRUE))
    }
  }

  if (requireNamespace("rstudioapi", quietly = TRUE)) {
    editor_path <- tryCatch(
      rstudioapi::getSourceEditorContext()$path,
      error = function(e) ""
    )
    if (nzchar(editor_path) && file.exists(editor_path)) {
      return(normalizePath(editor_path, winslash = "/", mustWork = TRUE))
    }
  }

  wd_candidate <- file.path(getwd(), "1-run.R")
  if (file.exists(wd_candidate)) {
    return(normalizePath(wd_candidate, winslash = "/", mustWork = TRUE))
  }

  stop(
    "Unable to determine the path to 1-run.R. ",
    "Run with `source('/absolute/path/to/1-run.R')` ",
    "or set working directory to the repo root before `source('1-run.R')`."
  )
}

resolve_repo_path <- function(path, repo_root) {
  if (grepl("^(/|~|[A-Za-z]:[/\\\\])", path)) {
    return(normalizePath(path, winslash = "/", mustWork = TRUE))
  }

  normalizePath(file.path(repo_root, path), winslash = "/", mustWork = TRUE)
}

repo_root <- tryCatch(
  dirname(get_this_file()),
  error = function(e) normalizePath(getwd(), winslash = "/", mustWork = TRUE)
)

run <- function(
    conda_env = "r-pygame",
    script = "python/ts.py",
    extra_args = NULL
) {
  script <- resolve_repo_path(script, repo_root)
  
  # Load reticulate + activate env
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("Package 'reticulate' is required but not installed.")
  }
  reticulate::use_condaenv(conda_env, required = TRUE)
  
  # Python binary
  py_bin <- reticulate::py_config()$python
  message("Using Python: ", py_bin)
  
  # Base args: always the script
  args <- c(script)
  
  # Optional passthrough arguments
  if (!is.null(extra_args)) {
    stopifnot(is.character(extra_args))
    args <- c(args, extra_args)
  }
  
  # Run the sim
  old_wd <- getwd()
  on.exit(setwd(old_wd), add = TRUE)
  setwd(repo_root)
  status <- system2(py_bin, args)
  invisible(status)
}
 
# Run
run()
