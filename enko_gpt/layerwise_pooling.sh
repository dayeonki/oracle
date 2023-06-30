python3 layerwise_pooling.py -s data/aihub/parts/enko.en.train.00 -t data/aihub/parts/enko.ko.train.00 -sl 0 -tl 1 -o data/layerwised/enko.en.train.00. -d 0 -b 20;
python3 layerwise_pooling.py -s data/aihub/parts/enko.en.train.01 -t data/aihub/parts/enko.ko.train.00 -sl 0 -tl 1 -o data/layerwised/enko.en.train.00. -d 1 -b 20;
python3 layerwise_pooling.py -s data/aihub/parts/enko.en.train.02 -t data/aihub/parts/enko.ko.train.00 -sl 0 -tl 1 -o data/layerwised/enko.en.train.00. -d 2 -b 20;
python3 layerwise_pooling.py -s data/aihub/parts/enko.ko.train.03 -t data/aihub/parts/enko.en.train.00 -sl 0 -tl 1 -o data/layerwised/enko.en.train.00. -d 3 -b 20;
python3 layerwise_pooling.py -s data/aihub/parts/enko.ko.train.04 -t data/aihub/parts/enko.en.train.00 -sl 0 -tl 1 -o data/layerwised/enko.en.train.00. -d 4 -b 20;
python3 layerwise_pooling.py -s data/aihub/parts/enko.ko.train.05 -t data/aihub/parts/enko.en.train.00 -sl 0 -tl 1 -o data/layerwised/enko.en.train.00. -d 5 -b 20;
python3 layerwise_pooling.py -s data/aihub/parts/enko.en.valid.00 -t data/aihub/parts/enko.en.train.00 -sl 0 -tl 1 -o data/layerwised/enko.en.train.00. -d 6 -b 20;
python3 layerwise_pooling.py -s data/aihub/parts/enko.ko.valid.00 -t data/aihub/parts/enko.ko.train.00 -sl 0 -tl 1 -o data/layerwised/enko.en.train.00. -d 7 -b 20;