cd BoolQ
gdown https://drive.google.com/uc?id=1GBViK4o6Fm0L83zl9Atv2J2wdxvBnQV4
tar -xvf boolq_data_results.tar
python inference.py
cd ..
cd Cola
gdown https://drive.google.com/uc?id=1SqFO4E2M1qIIJHusubUb5r2dklLdvlME
unzip cola_dataset_results.zip
python inference.py
cd ..
cd Copa
gdown https://drive.google.com/uc?id=1QB66EebwWP2XhgZFweG00NIu3QIBZ-hh
tar -xvf copa_data_results.tar
python inference.py
cd ..
cd WIC
gdown https://drive.google.com/uc?id=1DUaUTTl-YAwhZQmTHaLVHsPmA64dyQ75
tar -xvf wic_model.tar
python inference.py
cd ..
python make_submission.py
