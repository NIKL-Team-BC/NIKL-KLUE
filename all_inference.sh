cd BoolQ
if [ ! -f ./boolq_data_results.tar ]; then
    gdown https://drive.google.com/uc?id=1OZaN9qVNFlEbk7GIpaoREcEObLttibFk
fi
tar -xvf boolq_data_results.tar
python inference.py
cd ..
cd Cola
gdown https://drive.google.com/uc?id=1SqFO4E2M1qIIJHusubUb5r2dklLdvlME
unzip cola_dataset_results.zip
python inference.py
cd ..
cd Copa
if [ ! -f ./copa_data_results.tar ]; then
    gdown https://drive.google.com/uc?id=1QB66EebwWP2XhgZFweG00NIu3QIBZ-hh
fi
tar -xvf copa_data_results.tar
python inference.py
cd ..
cd WIC
if [ ! -f ./wic_model.tar ]; then
    gdown https://drive.google.com/uc?id=1DUaUTTl-YAwhZQmTHaLVHsPmA64dyQ75
fi
tar -xvf wic_model.tar
python inference.py
cd ..
python make_submission.py
