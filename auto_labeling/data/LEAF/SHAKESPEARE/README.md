# Shakespeare dataset (from LEAF dataset) preprocessing

1. The url for shakespeare dataset changed
    1) from http to https
    2) from old to old/old
#wget http://www.gutenberg.org/files/100/old/1994-01-100.zip
   
curl -v -L -O https://www.gutenberg.org/files/100/old/old/1994-01-100.zip

2. Copy data/utils and data/shakespeare from leaf repo: https://github.com/TalwalkarLab/leaf/

3. Generate data
    #cd shakespeare
    #./preprocess.sh -s niid --sf 1.0 -k 0 -t sample -tf 0.8
    #./stats.sh

4. Preprocess data
   %copy 'language_utils.py' from leaf: models/utils/language_utils.py
   python preprocess.py   
