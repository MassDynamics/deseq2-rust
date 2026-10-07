# Generate the edgeR ANOVA runs with RLE, upperquartile and none norm (golden corpus audit) as
# gen_shrink_runs.sh does: this repo's harness copy bind-mounted over
# MDFlexiComparisons/data-raw/golden-corpus, into a scratch corpus. Only the new run dirs are
# then moved into ~/wd/md-count-golden-corpus/runs; its index.json and shared/ are not touched.
#   sh corpus/gen_edger_anova_norm_runs.sh
set -e
H=$(cd "$(dirname "$0")/corpus-harness" && pwd)
P=~/wd/md-repos/MDFlexiComparisons; C=~/wd/md-count-golden-corpus; S=$(mktemp -d /tmp/edger-norm-corpus.XXXX)
RUNS="count_edger_anova_norm_RLE count_edger_anova_norm_upperquartile count_edger_anova_norm_none"
mkdir -p "$H/public"  # mount point for the public datasets; removed below
mkdir -p $S/shared && cp -R $C/shared/airway $C/shared/count_synth $S/shared/
SHA=$(DEVELOPER_DIR=/Library/Developer/CommandLineTools git -C "$P" rev-parse HEAD)$(DEVELOPER_DIR=/Library/Developer/CommandLineTools git -C "$P" diff --quiet HEAD -- R || echo -dirty)
HARNESS=$(cd "$H" && shasum -a 256 *.R *.sh | shasum -a 256 | cut -c1-64)
IMG=${MD_R_IMAGE:-md-flexi-r45-local}
DIG=$(/usr/local/bin/docker image inspect $IMG --format '{{.Id}}')
/usr/local/bin/docker run --rm --platform linux/amd64 -e MD_CORPUS_KIND=count -e MD_FLEXI_SHA=$SHA -e MD_IMAGE_DIGEST=$DIG -e MD_HARNESS_SHA256=$HARNESS -e R_PROFILE_USER=/dev/null -e LANG=C.UTF-8 -e LC_COLLATE=C -e TZ=Etc/UTC -e MD_GOLDEN_CORPUS_DIR=/corpus -v $S:/corpus -v "$P":"$P":ro -v "$H":"$P/data-raw/golden-corpus":ro -v "$P/data-raw/golden-corpus/public":"$P/data-raw/golden-corpus/public":ro -w "$P" $IMG Rscript data-raw/golden-corpus/generate_golden_corpus.R $RUNS
rmdir "$H/public"
for r in $RUNS; do
  if [ -e "$C/runs/$r" ]; then echo "exists, not moved: $r"; else mv "$S/runs/$r" "$C/runs/$r"; echo "added $r"; fi
done
diff -rq $S/shared $C/shared && echo "shared/ unchanged"
echo "scratch corpus left at $S"
