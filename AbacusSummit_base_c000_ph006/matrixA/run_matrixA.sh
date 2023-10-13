#!/bin/bash

sim=$1
z=$2
Rf=$3
Nmesh=$4
qname=$5

cd ..
python calc_quantities.py $sim $Rf
python write_out_quantities.py $sim $z $Rf sdelta1 nabla2d1 G2

cd matrixA
python run_matrixA.py $sim $z $Rf $Nmesh $qname
python save_matrixA_nooverlap.py $sim $z $Rf $Nmesh $qname

