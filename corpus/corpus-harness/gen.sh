cd ~/wd/md-repos/MDFlexiComparisons; P=$PWD; C=~/wd/md-limma-golden-corpus
SHA=$(DEVELOPER_DIR=/Library/Developer/CommandLineTools git -C "$P" rev-parse HEAD)$(DEVELOPER_DIR=/Library/Developer/CommandLineTools git -C "$P" diff --quiet HEAD -- R || echo -dirty)
# Harness scripts are untracked in MDFlexiComparisons; their hash goes in every manifest (the
# raw public inputs are pinned by RAW_SHA256 inside build_public_datasets.R, which is hashed).
HARNESS=$(cd "$P/data-raw/golden-corpus" && shasum -a 256 *.R *.sh | shasum -a 256 | cut -c1-64)
DIG=$(/usr/local/bin/docker image inspect ${MD_R_IMAGE:-md-flexi-r45-limma368} --format '{{.Id}}')
/usr/local/bin/docker run --rm --platform linux/amd64 -e MD_FLEXI_SHA=$SHA -e MD_IMAGE_DIGEST=$DIG -e MD_HARNESS_SHA256=$HARNESS -e R_PROFILE_USER=/dev/null -e LANG=C.UTF-8 -e LC_COLLATE=C -e MD_GOLDEN_CORPUS_DIR=/corpus -v $C:/corpus -v "$P":"$P" -w "$P" ${MD_R_IMAGE:-md-flexi-r45-limma368} Rscript data-raw/golden-corpus/generate_golden_corpus.R "$@" 2>&1 | grep -E "Warning|UNEXPECTED|expected an error|Error" | tail -12
for i in "$@"; do /opt/homebrew/bin/python3 -c "
import json,sys; m=json.load(open('$C/runs/$i/manifest.json')); print('==', '$i', m.get('status'), m.get('n_rows'), repr(m.get('error'))[:160], '| warn:', (m.get('warning_text') or '')[:200])"; done
