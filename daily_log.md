# Daily Paper Reading Log

## Table of Contents

- [2022](#2022)
    - [2022/05](#202205)   


## 2022
### 2022/05

- **2022/05/06, Friday.**

    1.<u>CoCa: Contrastive Captioners are Image-TextFoundation Models.</u> [[PDF]](https://arxiv.org/pdf/2205.01917.pdf)  
    - Main Idea: unify the single-encoder, dual-encoder and encoder-decoder paradigms with a caption head on top of CLIP-style architecture(dual-encoder). Pretraining the model with contrastive and caption loss.  
    - Experiments: CoCa obtains 86.3% *zero-shot* top-1 accuracy on ImageNet-1K, crazy, data is all you need!!!
    <p align="center"> <img src='images/20220506_CoCa.png' align="center" height="200px"> </p>
    
    2.<u>Sequencer: Deep LSTM for Image Classification.</u> [[PDF]](https://arxiv.org/pdf/2205.01972.pdf)
    - Main Idea: Sequencer uses LSTM to model long-range depandencies. 
    - Experiments: Sequencer2D-L, with 54M parameters, realizes 84.6% top-1 accuracy on ImageNet-1K.
    - Take away message: inference is quite slow, more robust to resolution change.
    <p align="center"> <img src='images/20220506_Sequencer.png' align="center" height="250px"> </p>  

    3.<u>Video Extrapolation in Space and Time.</u> [[PDF]](https://arxiv.org/pdf/2205.02084.pdf) [[Website]](https://cs.stanford.edu/~yzzhang/projects/vest/)
    - Main Idea: unify the novel view synthesis and video prediction tasks into a "Video Extrapolation in Space and Time" framework.
    - Take away message: joint training with relevant tasks might improve the performance on all tasks.
    <p align="center"> <img src='images/20220506_VEST.png' align="center" height="150px"> </p> 
  
- **2022/05/07, Saturday.**

    1.<u> **Episodic Memory Question Answering. CVPR 2022, Oral**.</u> [[PDF]](https://arxiv.org/pdf/2205.01652.pdf)  [[Website]](https://samyak-268.github.io/emqa/)
    - Main Idea: answer the question by grounding the target in a egocentric video.
    - Take away message: a new task - Episodic Memory Question Answering (EMQA).
    <p align="center"> <img src='images/20220507_EMQA.png' align="center" height="200px"> </p>

- **2022/05/08, Sunday.**

Pending...




