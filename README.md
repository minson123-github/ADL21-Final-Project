# ADL21-Final Project

## Dataset Conversion
Here is an example.
```console
python3 data_conversion.py data-0614/data-0614/schema.json data-0614/data-0614/train data-0614/data-0614/dev multiWoZ_2_0
```
* ${1}: schema.json directory
* ${2}: training dialogues directory
* ${3}: evaluate dialogues directory
* ${4}: multiwoz 2.0 data directory, the directory of conversion result
After conversion finished, you can download multiwoz 2.0 dataset and copy `*_db.json` into your conversion directory by yourself.
Maybe you need to create an empty `testListFile.json` to ensure that your create data process of model can be finished. Here is an example.
```console
touch testListFile.json
```

TODO: Run trade-dst model or something else...
