# Robustness-of-DNN-against-a-novel-adversarial-technique-in-RGB-and-infrared-domain
DNN have proven to be very sensible to particular types of perturbations. Here, a novel adversarial technique, called Pixle, is tested in two different image domain: RGB and Infrared. Despite it's applicability in the real-world scenario, the latter is usually under-explored domain, so it could be interesting to test also the efficacy of the novel adversarial attacks against it. 
The experiments will be tested on three of the most recent DNN in both image domain:
- ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
- ResNeXt (https://arxiv.org/pdf/1611.05431.pdf)
- ViT (https://openreview.net/pdf?id=YicbFdNTTy)

The datasets used are:
- Benchmark datasets
  - **CIFAR-10**
  - **Tiny ImageNet**
- Aligned RGB-Infrared dataset 
  - **NIR-Scene** (https://www.epfl.ch/labs/ivrl/research/downloads/rgb-nir-scene-dataset/)
  
# Pixle
This novel adversarial technique it's an $L_{0}$-norm based attacks that bases its own mechanism on a simple re-arrangement of the image's pixels. This simple mechanism is strong enough to guarantee high efficacy and with the minimum amount of perturbation. Here below is shown an example of a *Successful attack*

![image](https://user-images.githubusercontent.com/56520161/201474498-17fbfdda-6d43-49ca-8bc6-a4d8f899c682.png)
![image](https://user-images.githubusercontent.com/56520161/201474589-502f9635-f9c4-4ab1-9d73-2ebab92bb894.png)

The image on the right is the one attacked by Pixle, while the image on the left is the original version. As it is possible to see, by only mapping 2 single pixels it is possible to mislead the model and make it predict the class wrongly. 
There are two types of Pixle:
- **Black-box Pixle**.
- **White-box Pixle** called **Wixle**.

Both version uses the same mechanism of mapping the pixels but in two different ways. 

## Black-box Pixle
This version bases the mapping mechanism on a simple random search for the best pattern that can fool the model. 
The algorithm works in the following way:
- For every **restart** Pixle compute the following operations for a specific amount of iterations.
  - It samples a random patch, whose dimensions depends on the value give in input. 
  - From that patch it randomly map the pixels to other location in the image and only save the intermediate result if its loss value is lower than the current best one. 
  - These operation will be repeated according to the number of iterations previously defined. 
  - The best result, out of all the iterations, will be saved as new starting point for the next restart.
Pixle will only stops when:
- No more restarts/iterations are left.
- An intermediate result is already misleading the model. 

## Wixle
The white-box version of Pixle exploits an additional information to compute the mapping's pixel: the gradient of the image with respect to the loss. This information quantify the importance of the pixels and it is used to evaluate a probability distribution through which the pixels will be mapped. The mechanism of restart/iterations is kept, the only things that changes is the mapping done using probabilities.
In this way:
- The pixels more likely to be sampled as **source** pixel (the ones that will be mapped) will be the one with the higher value of the gradient.
- The pixels more likely to be sampled as **destination** pixel (the ones that will be replaced by the source pixels) will be the one with the lower value of the gradient.

In this way Pixle maps the most important pixels to the place of the least important pixels so that the attacks are more accurate and fast. 

## Results
The attacks are tested on both domain and for every DNN.

![image](https://user-images.githubusercontent.com/56520161/201477169-60fc5ea5-f3bb-41d8-9993-2bd40e87ab35.png)

The tables shows the percentage of successuful attacks.

For every configuration, the attacks seems to be effective and the Infrared suffers more than the rgb the effect of the perturbations.

## Transferability 
It is possible to transfer a successful pattern from one domain to the other? YES, **but** with a condition.
- Have access to the model gradient informations.

![image](https://user-images.githubusercontent.com/56520161/201477274-f9746552-cd0f-4e52-89f4-5fd3557c49c7.png)

By exploiting the gradient information it is possible to transfer a successful attack with a rate between 35% and 40% in case of RGB --> Infrared and a rate between 20% and 35% in case of Infrared --> RGB. The results shows that the colour informations are mor important than the spatial information of the image. 

# Defences
Four techniques are proposed to improve the robustness of these models against these type of perturbation.
- **Fast adversarial training**
- **Detection through filters**
- **Mitigation of perturbations effect using random layers**
- **Removal of perturbations using generative models**

## Fast adversarial training
Pixle is quite heavy in terms of operations and resources needed to craft a successful attack, so a standard approach is not feasible. Here, i propose a fast approach that helps the model to learn how the attacks works (without being always successfully attacked) by providing image that can be light-perturbated. 
It works in the following way:
- During training, for every batch, an image is attacked with probability 0.5 with a light Wixle attack (1 restart, 1 iteration) with a random percentage of pixels moved between 5% and 40% of the total number of pixels.

![image](https://user-images.githubusercontent.com/56520161/201477695-b16315c0-7e47-415f-93eb-a8a292c915e9.png)

Average success rate strongly decreased. 

![image](https://user-images.githubusercontent.com/56520161/201477738-aa0dc422-c285-4430-9468-d37fc14a73d6.png)

Significant percentage increase of the queries needed to craft a successful attack. 

## Detection through filters
Here, the idea is to work on the predictions of filtered and non-filtered images. Usually the predictions between adversarial images and filtered adversarial images differs while, in case of clean images this does not happens. 
In this way, by using a measure of difference on the vector of the predictions it is possible to discriminate clean from adversarial images.
The procedure is formalized in this way:
- Once set a threshold, the input image is copied and filtered. 
- The predictions vector will be then computed with their corresponding difference measure ($L_{1} norm of their difference). 
- If this difference exceeds the threshold, the image is labeled as 'adversarial' and discarded.
- Otherwise is kept.

![image](https://user-images.githubusercontent.com/56520161/201478008-a7613470-175d-4df5-aadd-51cce313d375.png)

Success rate, significantly reduced. 

## Mitigation of perturbations effect using random layers
The main idea, here, is to apply 2 random operations of augmentation (resizing and padding) to reduce the effect of the attack. 
To do that, for every model, 2 layers are place to the bottom of the model so that, every time an image is send to the classifier, the layers will randomly resize and pad in order to reduce the effectiveness of the attack. 

![image](https://user-images.githubusercontent.com/56520161/201478145-7ba87929-ba00-4ef5-a81c-013eeb80ff36.png)

## Removal of perturbations using generative models
Here, the perturbations of the attacks are cleaned using generative models. Generative models are known to be capable of creating images, that's why they are used for tasks such as Super-resolution. This defense, instead uses this models to create a copy of the input that has no perturbations. To do that, the generative models are trained on both attacked and clean images in order to make the generator able to create images, from the attacks, that are the most similar to their corresponding clean version.

![image](https://user-images.githubusercontent.com/56520161/201478360-c2a1c324-f3c2-471e-86bb-e57f749fa6fe.png)

The results are impressive.

![image](https://user-images.githubusercontent.com/56520161/201478449-cdb91a61-026c-44df-a23f-d4b948e4664d.png)
