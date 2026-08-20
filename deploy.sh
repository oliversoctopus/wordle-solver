#!/bin/sh
# Builds site/ and uploads it to Vercel.
#
#     ./deploy.sh              to the live URL
#     ./deploy.sh --preview    to a throwaway URL, to look at before promoting
#
# One step rather than two, so there is no gap between building and deploying
# for a rebuild to be forgotten in.  The project link lives in site/.vercel/,
# which build_site.sh carries across a rebuild; `vercel link` writes it once.
#
# The Vercel project's Root Directory has to be "." for this, since the CLI
# uploads site/ itself -- the git integration used to be pointed inside the
# repo instead, and has to be off now that site/ is not committed.

set -e
cd "$(dirname "$0")"
export PATH="$HOME/.cargo/bin:$PATH"

# Pin this once you know a version that works: @latest re-fetches whatever is
# newest on every deploy, which is a thing that can break overnight.
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
npx --yes "$CLI" deploy site $TARGET
