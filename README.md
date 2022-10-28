# ArtBecomeHuman

### Link to Progress Report:
https://docs.google.com/document/d/1r_XVNKTlKX4AIG9H7Rr_2ILExxBuKFvZH4KvV88vbQg/edit?usp=sharing

### TODO:
1. Figure out why current model is so accurate on current data
    1. Did validation set leak into training set?
    2. Is there an obvious bias (ex. difference in style) between the ai and real sets?
2. yoink this 256x256 art dataset: https://github.com/liaopeiyuan/artbench
3. use stable diffusion img2img to get matching ai generated image dataset: https://github.com/CompVis/stable-diffusion
4. Retrain and test on new dataset
    1. Iterate hyperparameters and architecure
5. Analyse outcomes
    1. Gradcam on correct/incorrect ai/real images
    2. Find distinguishing features between ai and real images

![graph showing loss throughout training](example_images/loss_graph_10-20.png?raw=true)

AI Identified Image: <br />
![graph showing loss throughout training](example_images/ai_example_activations.png?raw=true)

Non-AI Identified Image: <br />
![graph showing loss throughout training](example_images/non-ai_example_activations.png?raw=true)
