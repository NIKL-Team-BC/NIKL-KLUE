cd BoolQ
pip install transformers
gdown https://drive.google.com/uc?id=1k-6W3bTqFVlSBBtsJ42kPcBdBCO7bQkD
tar -xvf boolq_data_results.tar
python inference.py
cd ..
cd Copa
gdown https://drive.google.com/uc?id=1PxwwOiYxKb7PUBVByX0LYwnXm9IY6lfh
tar -xvf copa_data_results.tar
python inference.py
cd ..
cd WIC
gdown https://drive.google.com/uc?id=12P_4-yxo0x9I32HFWFu_OCGbA-EJakK0
tar -xvf wic_model.tar
python inference.py
cd ..
cd Cola
pip install transformers==3.5.1
gdown https://drive.google.com/uc?id=1thXqYuEW0xdZZhOqq_fS9EyJCDQlYhli
unzip cola_dataset_results.zip
python inference.py
cd ..
python make_submission.py
