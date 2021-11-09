cd BoolQ
pip install transformers
gdown https://drive.google.com/uc?id=1GBViK4o6Fm0L83zl9Atv2J2wdxvBnQV4
tar -xvf boolq_data_results.tar
cp ../new_test/BoolQ.tsv ./boolq_data_results/data/SKT_BoolQ_Test.tsv
python inference.py
cd ..

cd Copa
gdown https://drive.google.com/uc?id=1QB66EebwWP2XhgZFweG00NIu3QIBZ-hh
tar -xvf copa_data_results.tar
cp ../new_test/CoPA.tsv ./copa_data_results/data/SKT_COPA_Test.tsv
python inference.py
cd ..

cd WIC
gdown https://drive.google.com/uc?id=1DUaUTTl-YAwhZQmTHaLVHsPmA64dyQ75
tar -xvf wic_model.tar
cp ../new_test/WiC.tsv ./Data/NIKL_SKT_WiC_Test.tsv
python inference.py
cd ..

cd Cola
pip install transformers==3.5.1
gdown https://drive.google.com/uc?id=1d-eIMrLZSxreeiE-vcO6lX5hjF70c6zS
unzip cola_dataset_results.zip
cp ../new_test/CoLA.tsv ./cola_data_results/data/NIKL_CoLA_test.tsv
python inference.py
cd ..

python make_submission.py
