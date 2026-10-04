# Sourced by lint, test and release. edge-rust is internal and cargo fetches rnum and edger-core
# from it, so every step that builds Rust needs a GitHub token. GH_TOKEN from the environment wins
# (a local run: `GH_TOKEN=$(gh auth token) scripts/test`); on CI it is read from SSM, as
# md-timecourse does.
if [ -z "${GH_TOKEN:-}" ]; then
  GH_TOKEN=$(aws ssm get-parameter \
    --name /services/github/dependencies \
    --with-decryption \
    --query Parameter.Value \
    --output text)
fi
export GH_TOKEN
[[ -z "$GH_TOKEN" ]] && echo "GH_TOKEN is not set and could not be read from /services/github/dependencies" && exit 1

# `docker run` arguments that make git inside the container (cargo uses it, git-fetch-with-cli)
# rewrite the ssh URLs to HTTPS with the token. The token is passed by name, so it stays out of
# the process list.
GIT_TOKEN_ARGS=(-e GH_TOKEN -e GIT_CONFIG_COUNT=1
  -e GIT_CONFIG_VALUE_0=ssh://git@github.com/MassDynamics/)
# The key carries the token, so it is set inside the container from GH_TOKEN.
GIT_TOKEN_SH='export GIT_CONFIG_KEY_0="url.https://x-access-token:${GH_TOKEN}@github.com/MassDynamics/.insteadOf"'
