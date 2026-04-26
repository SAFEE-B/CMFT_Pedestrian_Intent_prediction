
![架构](https://github.com/user-attachments/assets/5ab392fa-2546-473e-adac-270f22160717)
This repository provides an implementation of Multi-Context-Fusion-Transformer. The project supports training and evaluation on the JAAD and PIE datasets.
Requirements

All dependencies are listed in requirements.txt. To set up the environment, run:
```bash
pip install -r requirements.txt
```
Training

On JAAD dataset
```bash
python train_test.py -c config_files/nonvisual/NonVisualModel.yaml
```

On PIE dataset
```bash
python train_test.py -c config_files_pie/nonvisual/NonVisualModel.yaml
```
