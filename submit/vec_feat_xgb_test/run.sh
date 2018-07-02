#/bin/bash
cd editdistance-master && python setup.py install

cd -

python main_ensemble.py $1 $2
