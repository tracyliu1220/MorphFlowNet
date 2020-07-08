#! /bin/bash

cd networks/channelnorm_package
python3 setup.py build
sudo python3 setup.py install
cd -

cd networks/correlation_package
python3 setup.py build
sudo python3 setup.py install
cd -

cd networks/resample2d_package
python3 setup.py build
sudo python3 setup.py install
cd -
