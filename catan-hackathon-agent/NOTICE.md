# Third-party notice

This kit depends on **catanatron** by Bryan Collazo, released under GPL-3.0-or-later.

* Upstream: <https://github.com/bcollazo/catanatron>
* We do **not** redistribute catanatron. `requirements.txt` installs it from its GitHub
  repository, pinned to one commit.

`catan_kit/_patch.py` changes one catanatron function at run time, in memory only. It sorts the
list of bank and port trade offers so that the order depends on the contents and not on memory
layout. Nothing on disk is modified. The reason, and the measurements behind it, are in that
file's docstring.

Because this kit is built on a GPL-3.0 work, the kit is released under the same license. See
`LICENSE`.
