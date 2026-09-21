#!/bin/sh
# Builds site/ and uploads it to Vercel.
#
#     ./deploy.sh              to the live URL
#     ./deploy.sh --preview    to a throwaway URL, to look at before promoting
#
# One step, so a rebuild cannot be forgotten between building and deploying.
# The project link lives in site/.vercel/, which build_site.sh carries across
# a rebuild; `vercel link` writes it once.  The Vercel project's Root
# Directory has to be "." and its git integration off, since the CLI uploads
# site/ itself.

set -e
cd "$(dirname "$0")"
export PATH="$HOME/.cargo/bin:$PATH"

# The CLI only uploads what build_site.sh already produced, so its version has
# no bearing on what deploys.  VERCEL_CLI pins it if a release ever misbehaves.
CLI="${VERCEL_CLI:-vercel@latest}"

TARGET="--prod"
if [ "$1" = "--preview" ]; then
    TARGET=""
fi

./build_site.sh

if [ ! -d site/.vercel ]; then
    echo
    echo "site/.vercel is missing, so the CLI does not know which project this"
    echo "is -- run 'npx $CLI link' inside site/ once, then this again."
    exit 1
fi

echo
echo "uploading site/"
# one request alone refreshes an expired login; the deploy's concurrent
# requests race for the single-use refresh token and fail "Not authorized"
npx --yes "$CLI" whoami >/dev/null
npx --yes "$CLI" deploy site $TARGET
