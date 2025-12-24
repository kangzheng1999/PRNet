# PRNet implementation


If you find this project useful, please cite:

```
@article{kang2025prnet,
  title={PRNet: Parallel Reinforcement Network for two-view correspondence learning},
  author={Kang, Zheng and Lai, Taotao and Li, Zuoyong and Wei, Lifang and Chen, Riqing},
  journal={Knowledge-Based Systems},
  volume={310},
  pages={112978},
  year={2025},
  publisher={Elsevier}
}
```

## Requirements

Please use Python 3.6, opencv-contrib-python (3.4.0.12) and Pytorch (>= 1.7.0). Other dependencies should be easily installed through pip or conda.


## Example scripts

### Run the demo

For a quick start, clone the repo and download the pretrained model.
```bash
git clone https://github.com/SuhZhang/ConvMatch 
cd ConvMatch 
```
Then download the pretrained models from [here](https://drive.google.com/drive/folders/1JKuIWhMXe9ve3wRPb_xZmX37ZH1bxAC3).

Then run the feature matching with demo ConvMatch.

```bash
cd ./demo && python demo.py
```

### Generate training and testing data

First download YFCC100M dataset.
```bash
bash download_data.sh raw_data raw_data_yfcc.tar.gz 0 8
tar -xvf raw_data_yfcc.tar.gz
```

Download SUN3D testing (1.1G) and training (31G) dataset if you need.
```bash
bash download_data.sh raw_sun3d_test raw_sun3d_test.tar.gz 0 2
tar -xvf raw_sun3d_test.tar.gz
bash download_data.sh raw_sun3d_train raw_sun3d_train.tar.gz 0 63
tar -xvf raw_sun3d_train.tar.gz
```

Then generate matches for YFCC100M and SUN3D (only testing) with SIFT.
```bash
cd ../dump_match
python extract_feature.py
python yfcc.py
python extract_feature.py --input_path=../raw_data/sun3d_test
python sun3d.py
```
Generate SUN3D training data if you need by following the same procedure and uncommenting corresponding lines in `sun3d.py`.



### Test pretrained model

We provide the model trained on YFCC100M and SUN3D described in our AAAI paper. Run the test script to get similar results in our paper (the generated putative matches are different once regenerating the data).

```bash
cd ./test 
python test.py
```
You can change the default settings for test in `./test/config.py`.

### Train model on YFCC100M or SUN3D

After generating dataset for YFCC100M, run the tranining script.
```bash
cd ./core 
python main.py
```

You can change the default settings for network structure and training process in `./core/config.py`.

### Train with your own local feature or data 

The provided models are trained using SIFT. You had better retrain the model if you want to use ConvMatch with your own local feature, such as RootSIFT, SuperPoint and etc. 

You can follow the provided example scirpts in `./dump_match` to generate dataset for your own local feature or data.

## Acknowledgement
This code is borrowed from [ConvMatch](https://github.com/krbangzi/ConvMatch)  If using the part of code related to data generation, testing and evaluation, please cite these papers.

```
@inproceedings{zhang2019learning,
  title={Learning two-view correspondences and geometry using order-aware network},
  author={Zhang, Jiahui and Sun, Dawei and Luo, Zixin and Yao, Anbang and Zhou, Lei and Shen, Tianwei and Chen, Yurong and Quan, Long and Liao, Hongen},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  pages={5845--5854},
  year={2019}
}
@inproceedings{zhao2021progressive,
  title={Progressive correspondence pruning by consensus learning},
  author={Zhao, Chen and Ge, Yixiao and Zhu, Feng and Zhao, Rui and Li, Hongsheng and Salzmann, Mathieu},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={6464--6473},
  year={2021}
}
@inproceedings{zhang2023convmatch,
  title={ConvMatch: Rethinking Network Design for Two-View Correspondence Learning},
  author={Zhang, Shihua and Ma, Jiayi},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2023}
}
```
