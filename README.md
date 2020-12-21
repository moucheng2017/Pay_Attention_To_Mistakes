[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.md)

This repository contains a PyTorch implementation of the BMVC 2020 paper ["Learning To Pay Attention To Mistakes", 2020](https://www.bmvc2020-conference.com/assets/papers/0335.pdf). 

[Mou-Cheng Xu](https://moucheng2017.github.io/) is the developer of the code.

1. Download the whole repo including both the code and the datasets folder, compile your environment using bmvc2020_environment.yml file.
2. Use Run.py to run and debug.

## How to adapt the repo on your own datasets
1. An example of the folder structure is in datasets folder.
2. After you prepare the datasets, you can easily tune the interface in Run.py.
3. In ''network'' argument in Run.py, we provide different combinations of our model to be called: ''ERF_encoder_fp'', ''ERF_encoder_fn'', ''ERF_decoder_fp'', ''ERF_decoder_fn'', ''ERF_all_fp'', ''ERF_all_fn''. 
4. When you use any configurations including 'fn' in ''network'', please set the ''reverse'' flag as ''True''.
5. ''ERF_all_fn'' is very robust in the difficult testing setting, ''ERF_all_fp'' is the next best model. They both significally outperform baselines.

## More difficult testing in the code:
To test the generalisation of the trained model on unseen data, we add adversarial noises when we evaluate IoU and Hausdorff distance. This is to compensate the simplicity of the experimental settings (aka. binary segmentation).

## Potential future work
1. Extending the framework to multi-class.
2. The framework has a great potential in other tasks such as denoising, deblurring and semi-supervised learning.


## Citation
If you find the code or the README useful for your research, please cite our paper:
```
@article{LearnMistakeAttention,
  title={Learning To Pay Attention To Mistakes},
  author={Xu, Mou-Cheng and P. Neil, Oxtoby and C. Alexander, Daniel and Jacob, Joseph},
  journal={BMVC},
  year={2020},
}
```
