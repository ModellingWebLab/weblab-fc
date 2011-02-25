#!/bin/bash
#

cd ..
CELL_MODEL_TESTS_SOURCE=`pwd`
cd $CHASTE_TEST_OUTPUT

cp CellModelTests/*/*/*_s1s2_curve.eps $CELL_MODEL_TESTS_SOURCE/papers/pbmb_model_interactions/images
cp CellModelTests/*/*/*_ICaL_IV_curve.eps $CELL_MODEL_TESTS_SOURCE/papers/pbmb_model_interactions/images
