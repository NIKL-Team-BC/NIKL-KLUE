cd BoolQ
pip install transformers
if [ ! -f ./boolq_data_results.tar ]; then
    wget https://storage.googleapis.com/nikl/boolq_data_results.tar
fi
tar -xvf boolq_data_results.tar
python inference.py
cd ..
cd Copa
if [ ! -f ./copa_data_results.tar ]; then
    wget https://storage.googleapis.com/nikl/copa_data_results.tar
fi
tar -xvf copa_data_results.tar
python inference.py
cd ..
cd WIC
if [ ! -f ./wic_model.tar ]; then
    wget https://storage.googleapis.com/nikl/wic_model.tar
fi
tar -xvf wic_model.tar
python inference.py
cd ..
cd Cola
pip install transformers==3.5.1
if [ ! -f ./cola_dataset_results.zip ]; then
    wget https://storage.googleapis.com/nikl/cola_dataset_results.zip
fi
unzip cola_dataset_results.zip
python inference.py
cd ..
python make_submission.py