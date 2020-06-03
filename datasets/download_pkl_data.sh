#!/bin/bash
# This file download all datasets used on this repository in the format pkl of pandas - python.


mkdir -p capes/
mkdir -p books/
mkdir -p scielo/

# TESTSET BOOKS
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=177dFn67zhVXIGHakNUA-iGtpfb89fHs6' -O 

# TRAINSET CAPES
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1y7HJkVl7XFsl7ZPWDZJIuszTTHCmaqvo' -O capes/TRAINSET.pkl

# TESTSET CAPES
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Cfwm71XHjKZwvqEBH1b4PgLcaE4Xgkt-' -O capes/TESTSET.pkl


# TRAINSET Scielo

wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1KgKTN7Nod6_QM4hcjG7JtGH5j_9A8QcJ' -O scielo/TRAINSET.pkl

# TESTSET Scielo
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1QEwk4PVm0onJnuDZnmNH1I9ycS2-ztMF' -O scielo/TESTSET.pkl
