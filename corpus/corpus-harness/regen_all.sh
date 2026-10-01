set -e
bash "$(dirname "$0")/gen.sh" > /tmp/regen-all.log 2>&1
cd ~/wd/md-repos/MDFlexiComparisons; P=$PWD; C=~/wd/md-limma-golden-corpus
/usr/local/bin/docker run --rm --platform linux/amd64 -e R_PROFILE_USER=/dev/null -e LANG=C.UTF-8 -e LC_COLLATE=C -e MD_GOLDEN_CORPUS_DIR=/corpus -v $C:/corpus -v "$P":"$P" -w "$P" ${MD_R_IMAGE:-md-flexi-r45-limma368} Rscript data-raw/golden-corpus/export_csv.R runs shared > /tmp/regen-export.log 2>&1
echo done
