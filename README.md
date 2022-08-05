# Explore-And-Match

Implementation of "[Explore-And-Match](https://arxiv.org/abs/2201.10168)".

## Getting Started
:warning: **Dependencies**:
- `cuda == 10.2`
- `torch == 1.8.0`
- `torchvision == 0.9.0`
- `python == 3.8.11`
- `numpy == 1.20.3`

### Dataset Preparation
- [ActivityNet](https://ieeexplore.ieee.org/document/7298698)
- [Charades](http://ai2-website.s3.amazonaws.com/publications/hollywood-homes.pdf)

split
- ActivityNet (train/val/test)
- Charades (train/test: 5338/1334)


#### Download ActivityNet
- **Request videos**: you need to request download with below form

	https://docs.google.com/forms/d/e/1FAIpQLSeKaFq9ZfcmZ7W0B0PbEhfbTHY41GeEgwsa7WobJgGUhn4DTQ/viewform

merge 'v1-2' and 'v1-3' into a single folder 'videos'.

- **Download annotations ([ActivityNet Captions](https://arxiv.org/abs/1705.00754))**

	https://cs.stanford.edu/people/ranjaykrishna/densevid/captions.zip

#### Download Charades
- **Download videos**
	
	https://prior.allenai.org/projects/charades

- **Download annotations** 

	https://github.com/jiyanggao/TALL


### Pre-trained features
- C3D
- CLIP


#### Preprocess
Get 64/128/256 frames per video:
```
bash preprocess/get_constant_frames_per_video.sh
```

### Extract features with CLIP
change 'val_1' to 'val' and 'val_2' to 'test'
CLIP encodings
```
bash preprocess/get_clip_features.sh
```

### Train
`activitynet`, `charades`
```
bash train_{dataset}.sh
```

### Evaluation
```
bash test_{dataset}.sh
```

### Configurations
refer to `lib/configs.py`


## Citation

	@article{woo2022explore,
	  title={Explore and Match: End-to-End Video Grounding with Transformer},
	  author={Woo, Sangmin and Park, Jinyoung and Koo, Inyong and Lee, Sumin and Jeong, Minki and Kim, Changick},
	  journal={arXiv preprint arXiv:2201.10168},
	  year={2022}
	}
