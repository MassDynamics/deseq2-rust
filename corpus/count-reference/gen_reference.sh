# Run the standalone edgeR / DESeq2 reference (PR-E2) over the count corpus, in the
# same pinned R image and environment as corpus-harness/gen_count.sh.
#   sh count-reference/gen_reference.sh [run_id ...]
R=$(cd "$(dirname "$0")" && pwd)
F=~/wd/md-repos/MDFlexiComparisons; C=~/wd/md-count-golden-corpus
IMG=${MD_R_IMAGE:-md-flexi-r45-local}
DIG=$(/usr/local/bin/docker image inspect $IMG --format '{{.Id}}')
# Stamped into every reference.json and reference/index.json: the md-limma commit (-dirty when
# count-reference/ has uncommitted changes) and this directory's hash, computed as gen_count.sh does.
G=/opt/homebrew/bin/git
SHA=$($G -C "$R" rev-parse HEAD)$($G -C "$R" diff --quiet HEAD -- . || echo -dirty)
REF=$(cd "$R" && shasum -a 256 *.R *.sh | shasum -a 256 | cut -c1-64)
/usr/local/bin/docker run --rm --platform linux/amd64 -e R_PROFILE_USER=/dev/null -e LANG=C.UTF-8 -e LC_COLLATE=C -e TZ=Etc/UTC -e MD_FLEXI_DIR="$F" -e MD_COUNT_CORPUS_DIR=/corpus -e MD_LIMMA_SHA=$SHA -e MD_REFERENCE_SHA256=$REF -e MD_IMAGE_DIGEST=$DIG -v $C:/corpus -v "$F":"$F":ro -v "$R":"$R":ro -w "$F" $IMG Rscript "$R/run_reference.R" "$@"
