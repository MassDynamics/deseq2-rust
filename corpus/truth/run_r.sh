# Run the R reference (run_r.R) over the truth corpus in the production R image.
#   sh corpus/truth/run_r.sh [scenario ...]
H=$(cd "$(dirname "$0")" && pwd)
F=${MD_FLEXI_DIR:-$HOME/wd/md-repos/MDFlexiComparisons}
T=${MD_TRUTH_DIR:-$HOME/wd/md-count-truth-corpus}
IMG=${MD_R_IMAGE:-md-flexi-r45-local}
exec /usr/local/bin/docker run --rm --platform linux/amd64 -e R_PROFILE_USER=/dev/null -e LANG=C.UTF-8 \
  -e LC_COLLATE=C -e TZ=Etc/UTC -e MD_FLEXI_DIR="$F" -e MD_TRUTH_DIR=/truth \
  -v "$T":/truth -v "$F":"$F":ro -v "$H":"$H":ro -w "$F" "$IMG" Rscript "$H/run_r.R" "$@"
