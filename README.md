# ADL21-Final Project

## Dataset Conversion
Here is an example.
```console
python3 data_conversion.py \
	-schema=data-0614/data-0614/schema.json \
	-train=data-0614/data-0614/train \
	-eval=data-0614/data-0614/dev \
	-test_seen=data-0614/data-0614/test_seen \
	-test_unseen=data-0614/data-0614/test_unseen \
	-output=multiWoZ_2_0
```
You can run `python3 data_conversion.py -h` to obtain description for each argument.
After conversion finished, you can download multiwoz 2.0 dataset and copy `*_db.json` into your conversion directory by yourself.
Maybe you need to create an empty `testListFile.json` to ensure that your create data process of model can be finished. Here is an example.
```console
touch testListFile.json
```

TODO: Run trade-dst model or something else...
