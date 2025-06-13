 docker build -t llava-temporal:latest .
[+] Building 0.2s (2/2) FINISHED                                                                                                                                                                                                          docker:default
 => [internal] load build definition from Dockerfile                                                                                                                                                                                                0.0s
 => => transferring dockerfile: 2.15kB                                                                                                                                                                                                              0.0s
 => ERROR [internal] load metadata for us-docker.pkg.dev/deeplearning-platform-release/gcp-deeplearning/common-cu121:latest                                                                                                                         0.2s
------
 > [internal] load metadata for us-docker.pkg.dev/deeplearning-platform-release/gcp-deeplearning/common-cu121:latest:
------
Dockerfile:4
--------------------
   2 |     #  Base image: Google Deep Learning VM CUDA 12.1 (Debian 11, Python 3.10)
   3 |     # ──────────────────────────────────────────────────────────────────────────────
   4 | >>> FROM us-docker.pkg.dev/deeplearning-platform-release/gcp-deeplearning/common-cu121:latest
   5 |     
   6 |     # 1. System build tools
--------------------
ERROR: failed to solve: us-docker.pkg.dev/deeplearning-platform-release/gcp-deeplearning/common-cu121:latest: failed to resolve source metadata for us-docker.pkg.dev/deeplearning-platform-release/gcp-deeplearning/common-cu121:latest: failed to authorize: failed to fetch anonymous token: unexpected status from GET request to https://us-docker.pkg.dev/v2/token?scope=repository%3Adeeplearning-platform-release%2Fgcp-deeplearning%2Fcommon-cu121%3Apull&service=us-docker.pkg.dev: 403 Forbidden
(base) serkanakin@seko-l4:~/projects/reproduction-project$ 