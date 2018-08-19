mkdir processed
mkdir processed/emotion
mkdir processed/motivation

python scripts/make_dataloader.py memory
python scripts/make_dataloader.py neural

python scripts/make_gendataloader.py memory
python scripts/make_gendataloader.py neural