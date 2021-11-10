cd BoolQ
pip install transformers
if [ ! -f ./boolq_data_results.tar ]; then
    wget https://storage.googleapis.com/nikl/boolq_data_results.tar
fi
tar -xvf boolq_data_results.tar
cp ../new_test/BoolQ.tsv ./boolq_data_results/data/SKT_BoolQ_Test.tsv
python inference.py
cd ..

cd Copa
if [ ! -f ./copa_data_results.tar ]; then
    wget https://storage.googleapis.com/nikl/copa_data_results.tar
fi
tar -xvf copa_data_results.tar
cp ../new_test/CoPA.tsv ./copa_data_results/data/SKT_COPA_Test.tsv
python inference.py
cd ..

cd WIC
if [ ! -f ./wic_model.tar ]; then
    wget https://storage.googleapis.com/nikl/wic_model.tar
fi
tar -xvf wic_model.tar
cp ../new_test/WiC.tsv ./Data/NIKL_SKT_WiC_Test.tsv
python inference.py
cd ..

cd Cola
pip install transformers==3.5.1
if [ ! -f ./cola_dataset_results.zip ]; then
    wget https://storage.googleapis.com/nikl/cola_dataset_results.zip
fi
unzip cola_dataset_results.zip
cp ../new_test/CoLA.tsv ./cola_data_results/data/NIKL_CoLA_test.tsv
python inference.py
cd ..

python make_submission.py
