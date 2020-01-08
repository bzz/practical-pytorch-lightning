#!/bin/bash
set -x

# fetching data from https://gist.github.com/bzz/bf4091329d2545b862dc3a6bb0537e65
if ! stat -t data/*.txt >/dev/null 2>&1 
then
  wget https://gist.github.com/bzz/bf4091329d2545b862dc3a6bb0537e65/raw/60b048765ff8a4ce702a73a6e795fc9972e70d4b/ru.txt
  wget https://gist.github.com/bzz/bf4091329d2545b862dc3a6bb0537e65/raw/60b048765ff8a4ce702a73a6e795fc9972e70d4b/us.txt
  mkdir -p data
  mv *.txt data
fi
