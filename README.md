# ArtBecomeHuman

![graph showing loss throughout training](loss_graph_10-20.png?raw=true)

### TODO:
1. Figure out why current model is so accurate on current data
    1. Did validation set leak into training set?
    2. Is there an obvious bias (ex. difference in style) between the ai and real sets?
2. grab this 256x256 art dataset: https://github.com/liaopeiyuan/artbench
3. use stable diffusion img2img to get matching ai generated image dataset: https://github.com/CompVis/stable-diffusion
4. Retrain and test on new dataset
    1. Iterate hyperparameters and architecure
5. Analyse outcomes
    1. Gradcam on correct/incorrect ai/real images
    2. Find distinguishing features between ai and real images

### Training Loss Graph:
![graph showing loss throughout training](example_images/loss_graph_10-20.png?raw=true)

### Example GradCam Feature Maps
(To generate more of these, run the visualizer.py script) <br />
>>>>>>> 689647a486be120c683f1c51883b9518dace9ff0

AI Identified Image: <br />
![graph showing loss throughout training](example_images/ai_example_activations.png?raw=true)

Non-AI Identified Image: <br />
![graph showing loss throughout training](example_images/non-ai_example_activations.png?raw=true)
