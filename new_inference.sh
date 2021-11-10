cd BoolQ
pip install transformers
if [ ! -f ./boolq_data_results.tar ]; then
    gdown https://drive.google.com/uc?id=1OZaN9qVNFlEbk7GIpaoREcEObLttibFk
fi
tar -xvf boolq_data_results.tar
cp ../new_test/BoolQ.tsv ./boolq_data_results/data/SKT_BoolQ_Test.tsv
python inference.py
cd ..

cd Copa
if [ ! -f ./copa_data_results.tar ]; then
    gdown https://drive.google.com/uc?id=1QB66EebwWP2XhgZFweG00NIu3QIBZ-hh
fi
tar -xvf copa_data_results.tar
cp ../new_test/CoPA.tsv ./copa_data_results/data/SKT_COPA_Test.tsv
python inference.py
cd ..

cd WIC
if [ ! -f ./wic_model.tar ]; then
    gdown https://drive.google.com/uc?id=1Rm1J6vYpZ09OTOVBBC0Un-gFTpcSoAch
fi
tar -xvf wic_model.tar
cp ../new_test/WiC.tsv ./Data/NIKL_SKT_WiC_Test.tsv
python inference.py
cd ..

cd Cola
pip install transformers==3.5.1
if [ ! -f ./cola_dataset_results.zip ]; then
    gdown https://drive.google.com/uc?id=1d-eIMrLZSxreeiE-vcO6lX5hjF70c6zS
fi
unzip cola_dataset_results.zip
cp ../new_test/CoLA.tsv ./cola_data_results/data/NIKL_CoLA_test.tsv
python inference.py
cd ..

python make_submission.py