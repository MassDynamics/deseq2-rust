# edgeR / DESeq2 count corpus (phase 2). Same container recipe as gen.sh, own corpus dir.
# MD_R_IMAGE must be an image whose package versions match the deployed MDFlexiComparisons
# image (plan G0); the manifest records what was actually loaded.
cd ~/wd/md-repos/MDFlexiComparisons; P=$PWD; C=~/wd/md-count-golden-corpus; mkdir -p $C
SHA=$(DEVELOPER_DIR=/Library/Developer/CommandLineTools git -C "$P" rev-parse HEAD)$(DEVELOPER_DIR=/Library/Developer/CommandLineTools git -C "$P" diff --quiet HEAD -- R || echo -dirty)
HARNESS=$(cd "$P/data-raw/golden-corpus" && shasum -a 256 *.R *.sh | shasum -a 256 | cut -c1-64)
IMG=${MD_R_IMAGE:-md-flexi-r45-local}
DIG=$(/usr/local/bin/docker image inspect $IMG --format '{{.Id}}')
/usr/local/bin/docker run --rm --platform linux/amd64 -e MD_CORPUS_KIND=count -e MD_FLEXI_SHA=$SHA -e MD_IMAGE_DIGEST=$DIG -e MD_HARNESS_SHA256=$HARNESS -e R_PROFILE_USER=/dev/null -e LANG=C.UTF-8 -e LC_COLLATE=C -e TZ=Etc/UTC -e MD_GOLDEN_CORPUS_DIR=/corpus -v $C:/corpus -v "$P":"$P" -w "$P" $IMG Rscript data-raw/golden-corpus/generate_golden_corpus.R "$@"
