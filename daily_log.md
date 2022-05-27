# Daily Paper Reading Log

## Table of Contents

- [2022](#2022)
    - [2022/05](#202205)   


## 2022
### 2022/05

- **2022/05/06, Friday.**

    1. <u>CoCa: Contrastive Captioners are Image-TextFoundation Models.</u> [[PDF]](https://arxiv.org/pdf/2205.01917.pdf)  
    - Main Idea: unify the single-encoder, dual-encoder and encoder-decoder paradigms with a caption head on top of CLIP-style architecture(dual-encoder). Pretraining the model with contrastive and caption loss.  
    - Experiments: CoCa obtains 86.3% *zero-shot* top-1 accuracy on ImageNet-1K, crazy, data is all you need!!!
    <p align="center"> <img src='images/20220506_CoCa.png' align="center" height="200px"> </p>
    
    2. <u>Sequencer: Deep LSTM for Image Classification.</u> [[PDF]](https://arxiv.org/pdf/2205.01972.pdf)
    - Main Idea: Sequencer uses LSTM to model long-range depandencies. 
    - Experiments: Sequencer2D-L, with 54M parameters, realizes 84.6% top-1 accuracy on ImageNet-1K.
    - Take away message: inference is quite slow, more robust to resolution change.
    <p align="center"> <img src='images/20220506_Sequencer.png' align="center" height="250px"> </p>  

    3. <u>Video Extrapolation in Space and Time.</u> [[PDF]](https://arxiv.org/pdf/2205.02084.pdf) [[Website]](https://cs.stanford.edu/~yzzhang/projects/vest/)
    - Main Idea: unify the novel view synthesis and video prediction tasks into a "Video Extrapolation in Space and Time" framework.
    - Take away message: joint training with relevant tasks might improve the performance on all tasks.
    <p align="center"> <img src='images/20220506_VEST.png' align="center" height="150px"> </p> 
  
- **2022/05/07, Saturday.**

    4. <u> **Episodic Memory Question Answering. CVPR 2022, Oral**.</u> [[PDF]](https://arxiv.org/pdf/2205.01652.pdf)  [[Website]](https://samyak-268.github.io/emqa/)
    - Main Idea: answer the question by grounding the target in a egocentric video.
    - Take away message: a new task - Episodic Memory Question Answering (EMQA).
    <p align="center"> <img src='images/20220507_EMQA.png' align="center" height="200px"> </p>

- **2022/05/08, Sunday.**

    5. <u> Improving Visual Grounding with Visual-Linguistic Verification and Iterative Reasoning. CVPR 2022.</u> Yang, Li, et al. CAS. [[PDF]](https://arxiv.org/pdf/2205.00272.pdf) [[Code]](https://github.com/yangli18/VLTVG)
    - Main Idea: better multi-modal fusion strategy with pixel-wise correlation and multi-stage cross-modal decoder.
    - Take away message: fusion is very important for multi-modal learning.
    <p align="center"> <img src='images/20220508_VLTVG.png' align="center" height="250px"> </p>

    6. <u> COOPERNAUT: End-to-End Driving with Cooperative Perception for Networked Vehicles. CVPR 2022.</u> Cui, Jiaxun, et al. The University of Texas at Austin. [[PDF]](https://arxiv.org/pdf/2205.02222.pdf) [[Code]](https://github.com/UT-Austin-RPL/Coopernaut) [[Website]](https://ut-austin-rpl.github.io/Coopernaut/)
    - Main Idea: incorporate the visual information (point-cloud) from other vehicles to improve the success rate in challenging driving situations.
    - Experiments: 40% improvement in average success rate over egocentric driving models in these challenging driving situations.
    - Take away message: a interesting task.
    <p align="center"> <img src='images/20220508_COOP.png' align="center" height="250px"> </p>

    7. <u> Language Models Can See: Plugging Visual Controls in Text Generation. Submit to NeurIPS 2022.</u> Sun, Yixuan, et al. University of Cambridge. [[PDF]](https://arxiv.org/pdf/2205.02655.pdf) [[Code]](https://github.com/yxuansu/MAGIC)
    - Main Idea: combines an off-the-shelf LM (i.e., GPT-2) and an image-text matching model (i.e., CLIP) for image-grounded text generation.
    - Take away message: previous works mainly use CLIP to generate images with the guide of provided language, this work uses CLIP in a opposite way.
    <p align="center"> <img src='images/20220508_MAGIC.gif' align="center" height="250px"> </p>

    8. <u> All You May Need for VQA are Image Captions.</u>  Changpinyo, Soravit, et al. Google. [[PDF]](https://arxiv.org/pdf/2205.01883.pdf)
    - Main Idea: generate questions from the image caption.
    - Take away message: none.
    <p align="center"> <img src='images/20220508_VQA.png' align="center" height="250px"> </p>
    
- **2022/05/09, Monday.**

    9. <u> Inferring and Executing Programs for Visual Reasoning. ICCV 2017.</u> Johnson, Justin, et al. UC Berkeley. [[PDF]](https://arxiv.org/pdf/1705.03633.pdf) [[Code]](https://github.com/yangli18/VLTVG)
    - Main Idea: Specialize neural modules to execute different kind visual reasoning.
    <p align="center"> <img src='images/20220509_INFER.png' align="center" height="250px"> </p>

- **2022/05/10, Tuesday.**

    10. <u> SwapMix: Diagnosing and Regularizing the Over-Reliance on Visual Context in Visual Question Answering. CVPR 2022.</u> Gupta, Vipul, et al. U Penn. [[PDF]](https://arxiv.org/pdf/2204.02285.pdf) [[Code]](https://github.com/vipulgupta1011/swapmix)
    - Main Idea: Investigate the over-reliance on unrelate visual content for VQA models. 
    - Methods: Replace the unrelated visual content with similar objects and test the performance of VQA models. Also, the replacing can also be applied as a data augmentation which boost the robustness of VQA models.
    - Take away message: the robustness of multi-modal models can be impacted by visual content.
    <p align="center"> <img src='images/20220510_SwapMix.png' align="center" height="250px"> </p>

    11. <u> Words Aren’t Enough, Their Order Matters: On the Robustness of Grounding Visual Referring Expressions. ACL 2020.</u> Walmer, Matthew, et al. U Maryland. [[PDF]](https://arxiv.org/pdf/2112.07668.pdf) [[Code]](https://github.com/SRI-CSL/TrinityMultimodalTrojAI)
    - Main Idea: models do not change their answer when the order of words in a sentence changed.
    
    12. <u> Visual Spatial Reasoning. arXiv 2022/4/30.</u> Liu, Fangyu, et al. U Cambridge. [[PDF]](https://arxiv.org/pdf/2205.00363.pdf) [[Code]](https://github.com/cambridgeltl/visual-spatial-reasoning)
    - Main Idea: Build a 2D visual spatial reasoning(VSR) dataset to investigate the visual spatial reasoning ability of VL models. 
    - Dataset details:
        - 65 spatial relations and has more than 10k(10,119) data points, using 6940 image from MSCOCO.
    - Take away message: 
        - positional encodings are extremely important for VSR task.
        - models' by-relation performance barely correlates with the number of training examples.
        - a large gap between human and model performance (human ceiling on the VSR task is above 95% and models only achieve around 70%).

    13. <u> **Dual-Key Multimodal Backdoors for Visual Question Answering. CVPR 2022**.</u> Walmer, Matthew, et al. U Maryland. [[PDF]](https://arxiv.org/pdf/2112.07668.pdf) [[Code]](https://github.com/SRI-CSL/TrinityMultimodalTrojAI)
    - Main Idea: Training the VQA models with multimodal backdoors. The backdoor will only be activated when triggers in both modalities are present. 
    - Take away message: TrojVQA is a large collection of clean and trojan VQA models to enable research in defending against multimodal backdoors.
    <p align="center"> <img src='images/20220510_TrojVQA.png' align="center" height="250px"> </p>

- **2022/05/14, Saturday.**

    14. <u> Making Better Mistakes: Leveraging Class Hierarchies with Deep Networks.</u> Bertinetto, Luca, et al. Five AI. [[PDF]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Bertinetto_Making_Better_Mistakes_Leveraging_Class_Hierarchies_With_Deep_Networks_CVPR_2020_paper.pdf) [[Code]](https://github.com/fiveai/making-better-mistakes)
    - Main Idea: Leveraging class hierarchies to make better mistakes on image classification.
    - Take away message: Although top-1 acc is improving, the severity(stupidity) of mistake is fairly unchanged.

- **2022/05/15 - 2022/05/20, Rushing NeurIPS.**

- **2022/05/21, Saturday.**

    15. <u> **Open-Vocabulary Object Detection Using Captions. CVPR 2021 Oral**.</u> Zareian, Alireza, et al. U Columbia. [[PDF]](https://arxiv.org/pdf/2011.10678.pdf) [[Code]](https://github.com/alirezazareian/ovr-cnn)
    - Main Idea: Pre-train the visual backbone with image-text pairs which is easy to access from web to learn the visual semantics about diverse classes(include classes which do not have box annotation). 
    <p align="center"> <img src='images/20220521_OpenDetect.png' align="center" height="250px"> </p>

    16. <u> ScanQA: 3D Question Answering for Spatial Scene Understanding. CVPR 2022.</u> Azuma Daichi, Alireza, et al. U Columbia. [[PDF]](https://arxiv.org/pdf/2112.10482.pdf) [[Code]](https://github.com/ATR-DBI/ScanQA)
    - Main Idea: Introduce a 3D-QA dataset with question-answer pair and correspoinding box annotations.
    - Limitation: The annotations are derived from the ScanNet dataset using question generation models.
    - Notes:
        - Surprisingly, the randomimage+2D-QA models performs quite well.
    - Take away message: Object classification and localization both help the question-answering task.
    <p align="center"> <img src='images/20220521_ScanQA.png' align="center" height="250px"> </p>

- **2022/05/22, Sunday.**

    17. <u> Bottom-Up and Top-Down Attention for Image Captioning
    and Visual Question Answering. CVPR 2018 Oral.</u> Yu, Zhou, et al. Hangzhou Dianzi U. [[PDF]](https://arxiv.org/pdf/1707.07998.pdf) [[Code]](https://github.com/peteanderson80/bottom-up-attention)
    - Main Idea: Bottom-up attention with top-down attention. Bottom-up means purely visual feed-forward attention mechanisms. Top-down means task-specific attention, *e.g.*, given a query "looking for ball".
    - Method: Implementing the bottom-up attention by pre-training a FasterRCNN on Visual Genome dataset. The top-down attention is specifically designed for different tasks.
    - Notes: This a common strategy.

    18. <u> D3Net: A Speaker-Listener Architecture for Semi-supervised Dense Captioning and Visual Grounding in RGB-D Scans. arXiv 2021/12/2.</u> Chen, Zhenyu, et al. [[PDF]](https://arxiv.org/pdf/2112.01551.pdf) [[Code]](https://github.com/daveredrum/D3Net)
    - Main Idea: Design a speaker-listener architecture, speaker have to generate discriminative descriptions so that listener can better localize the described targets.
    - Method: Training by reinforce algorithm.
    - Take away message: Fine-tuning a 3D detector is essential to get good performance.
    <p align="center"> <img src='images/20220522_D3Net.png' align="center" height="250px"> </p>

    19. <u> Declaration-based Prompt Tuning for Visual Question Answering. IJCAI 2022.</u> Liu, Yuhang, et al. Huazhong University of Science and Technology. [[PDF]](https://arxiv.org/pdf/2205.02456.pdf) [[Code]](https://github.com/CCIIPLab/DPT)
    - Main Idea: Reformulate the VQA training objective to MLM and ITM, and then transform the question to a declaration type which both close the gap between the pre-training and fine-tuning.
    - Method: Training by reinforce algorithm.
    - Take away message: Similar objectives used in fine-tuning will help to transfer the knowledge learned in pre-training.
    <p align="center"> <img src='images/20220522_DPT.png' align="center" height="250px"> </p>

    20. <u> Answer-Me: Multi-Task Open-Vocabulary Visual Question Answering. arXiv 2022/5/2.</u> Piergiovanni, AJ, et al. Google Research. [[PDF]](https://arxiv.org/pdf/2205.00949.pdf) [Code Not Found]
    - Main Idea: Design a multi-task task-aware general framework which is able to deal with open-vocabulary VQA.
    - Method: Pre-train with a encoder-decoder architecture.
    - Take away message: semantically related tasks benefit from multi-task training as visual linguistic concepts are shared, helping their co-training.
    <p align="center"> <img src='images/20220522_AnswerMe.png' align="center" height="250px"> </p>

    21. <u> Rethinking Diversified and Discriminative Proposal Generation for Visual Grounding. IJCAI 2018.</u> Yu, Zhou, et al. Hangzhou Dianzi U. [[PDF]](https://arxiv.org/pdf/1805.03508.pdf) [[Code]](https://github.com/XiangChenchao/DDPN)
    - Main Idea: Generated proposals should be diversified and discriminative simultaneously: 1) the proposals of all images should be diversified to detect objects from open-vocabulary classes, and 2) the proposals of an individual image should be discriminative to guarantee that the proposals and visual features accurately represent the true semantic.
    - Method: Pretraining the object detector on Visual Genome dataset, emmmm.
    - Limitation: The story is attractive, however, the method is really have nothing to say.
    - Take away message: Soft label help the performance a lot, as show in Table 2.

- **2022/05/23, Monday.**

    22. <u> Weakly Supervised Relative Spatial Reasoning for Visual Question Answering. ICCV 2021.</u> Banerjee, Pratyay, et al. Arizona State University. [[PDF]](https://arxiv.org/pdf/2109.01934.pdf) [[Code]](https://github.com/pratyay-banerjee/weak_sup_vqa)
    - Main Idea: Incoporating the object centroid estimation(OCE) and relative position estimation(RPE) task with 2D Visual Question Answering task to enhance the spatial reasoning ability of visual model.
    - Take away message: Formulating the OCE and RPE as regression task will fail. In contrast, formulating them as bin-classification task will success!
    <p align="center"> <img src='images/20220523_VQA-SR.png' align="center" height="250px"> </p>

- **2022/05/24, Tuesday.** 

    23. <u> Probing the Role of Positional Information in Vision-Language Models. ACL submission 2022/01.</u> Anonymous. [[PDF]](https://openreviewnet/pdf?id=2Ubik08ztdB) [Code Not Found]
    - Main Idea: Probing the role of positional information in a unimodal way and a cross-modal way.
    - Take away message: 
        - current vision-language pre-training strategies fail to align the visual positional information with textual positional information.
        - replace the positional word in text with its counterpart(*e.g.*, left/right) and pre-train models to distinguish the text. 
        - object center is enough for representing the positional information for a object.

    24. <u> Things not Written in Text: Exploring Spatial Commonsense from Visual Signals. ACL 2022.</u> Liu, Xiao, et al. [[PDF]](https://aclanthology.org/2022.acl-long.168.pdf) [[Code]](https://github.com/xxxiaol/spatial-commonsense)
    - Take away message: 
        - spatial reasoning includes objects' scale, relative position between objects.
        - many works found that PLMs lack of spatial reasoning ability.
    
    25. <u> X-Trans2Cap: Cross-Modal Knowledge Transfer using Transformer for 3D Dense Captioning. CVPR 2022.</u> Yuan, Zhihao, et al. [[PDF]](https://arxiv.org/pdf/2203.00843.pdf) [[Code]](https://github.com/CurryYuan/X-Trans2Cap)
    - Main Idea: Incoporating 2D images into 3D dense caption task. To avoid using 2D modality in inference stage, authors propose to build a 3D single-modal model by distilling knowledge from a 2D-3D multi-modal network
    <p align="center"> <img src='images/20220524_XTrans2Cap.png' align="center" height="250px"> </p>

    26. <u> Pri3D: Can 3D Priors Help 2D Representation Learning?. ICCV 2021.</u> Hou, Ji, et al. [[PDF]](https://arxiv.org/pdf/2104.11225.pdf) [[Code]](https://github.com/Sekunde/Pri3D)
    - Main Idea: Pre-training with 3D point cloud data to help 2D tasks.
    - Method: View-invariant contrastive loss between 2D images and geometric prior contrastive loss between 2D and 3D.
    <p align="center"> <img src='images/20220524_Pri3D.png' align="center" height="250px"> </p>

- **2022/05/25, Wednesday.** 

    27. <u> Learning from 2D: Contrastive Pixel-to-Point Knowledge Transfer for 3D Pretraining (2D help 3D). arXiv 2021/04.</u> Liu, Yueh-Cheng, et al. National Taiwan University. [[PDF]](https://arxiv.org/pdf/2104.04687.pdf) [Code Not Found]
    - Main Idea: Pre-training with 2D images for 3D point cloud to help the 3D tasks.
    <p align="center"> <img src='images/20220525_LearnFrom2D.png' align="center" height="250px"> </p>

    28. <u> Self-Supervised Image Representation Learning with Geometric Set Consistency (3D help 2D). CVPR 2022.</u> Chen, Nenglun, et al. The University of Hong Kong. [[PDF]](https://arxiv.org/pdf/2203.15361.pdf) [Code Not Found]
    - Main Idea: The semantic cues lie in the discontinuities or depth gaps in 3D point cloud.
    - Method: 
        - Cluster the 3D point cloud into many local small segments, (geometric consistency sets). 
        - Propose a Set-InfoNCE loss to perform contrastive learning between point cloud geometric consistency set and images.
        - First, pre-training the model with Pri3D strategy. And then train it with proposed method.
    - Take away message: The gain comes from the semantic segmentation of 3D point cloud scene with a conventional method.
    <p align="center"> <img src='images/20220525_GeoConsist.png' align="center" height="250px"> </p>

    29. <u> Image-to-Lidar Self-Supervised Distillation for Autonomous Driving Data (2D help 3D). CVPR 2022.</u> Sautier, Corentin, et al. valeo.ai, Paris, France. [[PDF]](https://arxiv.org/pdf/2203.16258.pdf) [[Code]](https://github.com/valeoai/SLidR)
    - Main Idea: Similar to No.28, conduct contrastive learning between visually similar regions.
    - Method: 
    - Take away message: The design 2D projection head matters a lot. Stacking multiple convolution layers with kernel size larger than 1 will fail due to the leak of contextual information. 
    <p align="center"> <img src='images/20220525_Image2Lidar.png' align="center" height="250px"> </p>

- **2022/05/26, Thursday.** 

    30. <u> SimIPU: Simple 2D Image and 3D Point Cloud Unsupervised Pre-Training for Spatial-Aware Visual Representations (2D help 3D). AAAI 2022.</u> Li, Zhenyu, et al. HIT. [[PDF]](https://arxiv.org/pdf/2112.04680.pdf) [[Code]](https://github.com/zhyever/SimIPU)
    - Main Idea: 2D and 3D pre-training for outdoor lidar(point cloud data).
    - Method: 
        - Intra-modal(lidar) learning is performed by contrasting different views' features.  Use global feature other than local feature due to outdoor data is much more noise and massive.
        - Inter-modal(lidar-2D) learning is conducted by contrast paired 2D-3D features.
    - Take away message: Change the application domain of 2D-3D pre-training. Not interesting.
    <p align="center"> <img src='images/20220526_SimIPU.png' align="center" height="250px"> </p>

    31. <u> P4Contrast: Contrastive Learning with Pairs of Point-Pixel Pairs for RGB-D Scene Understanding. arXiv 2020/12.</u> Li, Zhenyu, et al. Xidian University. [[PDF]](https://arxiv.org/pdf/2012.13089.pdf) [Code Not Found]
    - Main Idea: Contrast with 
    - Method: 
    - Take away message: 
    <p align="center"> <img src='images/20220526_SimIPU.png' align="center" height="250px"> </p>

    32. <u> Contrastive Multimodal Fusion with TupleInfoNCE. ICCV 2021.</u> Liu, Yunze, et al. Xidian University. [[PDF]](https://arxiv.org/pdf/2107.02575.pdf) [Code Not Found]
    - Main Idea: (Similar to the idea of P4Contrast) Instead of contrasting different data modalities, we propose to contrast multimodal input tuples, where each tuple element corresponds to one modality. 
    <p align="center"> <img src='images/20220526_TupleInfoNCE.png' align="center" height="250px"> </p>


Pending...
