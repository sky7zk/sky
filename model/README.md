# Model files

Do not commit model weights to GitHub. Before local testing, run
`./prepare_model.sh`. The script copies the selected checkpoint and
normalization configuration into this directory and creates `model.tar.gz`.

Grand Challenge extracts the uploaded model archive under `/opt/ml/model`.

