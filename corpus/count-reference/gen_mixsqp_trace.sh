# Run trace_mixsqp.R in the pinned R image against <corpus>/reference-shrink/.
R=$(cd "$(dirname "$0")" && pwd)
C=${MD_COUNT_CORPUS_DIR:-~/wd/md-count-golden-corpus}
IMG=${MD_R_IMAGE:-md-flexi-r45-local}
/usr/local/bin/docker run --rm --platform linux/amd64 -e R_PROFILE_USER=/dev/null -e LANG=C.UTF-8 -e TZ=Etc/UTC -e MD_COUNT_CORPUS_DIR=/corpus -v $C:/corpus -v "$R":"$R":ro $IMG Rscript "$R/trace_mixsqp.R"
