
Performance of different versions (note that the original images we collected were ganked):

Legacy: Gets pretty much everything wrong
Low-Quality Trained: Cannot identify high quality AI images
Combined: Works for all stable-diffusion and artbench data. Also good with dall-e. However, cannot identify non-artbench real art.
Combined With noise: Works for all stable-diffusion and artbench data. Also good with dall-e, but slightly worse than combined. However, cannot identify non-artbench real art.

# ArtBecomeHuman

To run the ai-art generating scripts, put them in the /scripts/ folder of stable-diffusion and modify the constants at the top of the files.

### Updated TODO:
   1. The model is performing really well on our stable-diffusion/artbench set, but underperforming when given other images. Why? By visual inspection, it appears that the images we created are 'worse' (possibly due to prioritizing speed over quality, or bad prompts) than those we find on the internet. Also, they seem to have a very high color saturation compared to the human images. It could also be the invisible water marks on the other images throwing it off, but I doubt it.
   2. If the problem is low-quality images because of generation speed, I don't think we have time to make more high-quality ones on our own, so the solution is to again scrape images from the web. This raises the problem of invisible watermarks again, but I think similar to 5 this problem could be solved by putting the marks on some of our images. To get images, there is this database [https://lexica.art/docs](https://lexica.art/docs) with an api to get images based on prompt or revserse search (but I've already gotten rate-limited by them so this might not scale). We could also just do a google api call to search "stable diffusion painting of..." and then download the first image, but then our data will be less variable.
   3. If the problem is low quality prompts, I found a tool to convert images to prompts that I am running over the dataset (it will take a few hours). Then we can use the new prompts to see if we get better/more representative images.
   4. If the problem is color saturation (this is the best case), then we can tell by retraining the model using gray scale. This is a trivial task that I'll finish soon. If this does turn out to fix the generalization problem, then I think that would be a _significant_ result for us to talk about.
   5. If the problem is because we trained without watermarks but the other images do have them, I think the solution is to randomly put water marks on some percentage of our data (both ai and non-ai). Hopefully this will get the model accustomed to seeing them, while also ignoring them as an important feature because they are found for both ai and human labeled images.
   6. If none of these things work then I think having the result of "training a model on one type of ai images doesn't translate well to others" is still a significant finding that we can write about and try to justify.
    

### Training Loss Graph:
![graph showing loss throughout training](example_images/loss_graph_10-20.png?raw=true)

### Example GradCam Feature Maps
(To generate more of these, run the visualizer.py script) <br />
>>>>>>> 689647a486be120c683f1c51883b9518dace9ff0

AI Identified Image: <br />
![graph showing loss throughout training](example_images/ai_example_activations.png?raw=true)

Non-AI Identified Image: <br />
![graph showing loss throughout training](example_images/non-ai_example_activations.png?raw=true)
