# Tracked font assets

This directory contains Font Awesome-style `fa-brands-400`, `fa-regular-400`,
and `fa-solid-900` assets in several font formats (`eot`, `svg`, `ttf`, `woff`,
and `woff2`). They are tracked files, not part of the local model inference API.

No `BlockchainFont-*` or `TechMono-*` files are included, and this repository
does not provide the previously documented Blockchain Explorer UI or a
`public/index.html` font-loading contract. Applications that use these files
must supply their own styling and verify applicable font licensing.

The tracked workspace-startup safety tests also reject disguised executable
`*.lkf` files in this directory. That static check does not audit a host or
installed editor extensions.
