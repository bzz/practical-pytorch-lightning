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

# fetching CodeSearchNet + pre-built vocabulary
if ! stat -t data/codesearchnet >/dev/null 2>&1
then
  wget 'https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/java.zip'
  mkdir -p data/codesearchnet
  unzip java.zip
  mv java data/codesearchnet
  wget 'https://gist.github.com/bzz/cfa601b589893b4e709de98acdf0109f/raw/9dbfa3c086a1bf4ae745996242885f149f5e28d6/vocab.programming_function_name.8192.subwords'
fi
