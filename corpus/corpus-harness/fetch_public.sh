# Downloads the two raw public files into public/ and checks them against the SHA-256 that
# build_public_datasets.R also enforces. Run from the MDFlexiComparisons repo root, then run
# build_public_datasets.R in the image to write public/*.rds.
set -e
D=data-raw/golden-corpus/public; mkdir -p "$D"
curl -sSfo "$D/phospho.cells.Ins.pe.rda" \
  "https://raw.githubusercontent.com/PYangLab/PhosR/master/data/phospho.cells.Ins.pe.rda"
curl -sSfLo "$D/ST000001_AN000001.tsv" \
  "https://www.metabolomicsworkbench.org/rest/study/analysis_id/AN000001/datatable/"
shasum -a 256 -c - <<SUMS
ab577077fc8f1f0fad33f0fdd6b925c2b0c337f5a0bf827b4d36f5dc7832dc09  $D/phospho.cells.Ins.pe.rda
fe250e51fe7c04b27cc772e2069b1c3ab6932f0e019d3b471ff831ff160a3f13  $D/ST000001_AN000001.tsv
SUMS
