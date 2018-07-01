#/bin/bash
cd editdistance-master && python setup.py install

cd -

python main.py $1 $2
