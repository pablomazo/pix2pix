# pix2pix2pixel
[Spanish version](https://github.com/pablomazo/pix2pix/blob/master/README_ES.md)

## Summary.
PyTorch implementation of Pix2Pix architecture as propose in the [original article](https://arxiv.org/pdf/1611.07004.pdf).

Pix2Pix architecture is used to detect and pixel faces from a given image. The inverse process was also tried although results were not as satisfactory, maybe because the training set was too small of more training time is necessary.

## Dependencies
- PyTorch
- PIL
- numpy
- matplotlib
- torchvision

## Files
- pix2pix_faces: Files to execute training and visualization of results.
	- dataloader.py: Class to load and execute the requiered transformations on input and output images.
	- model.py: Generator and discriminator classes and function to save checkpoints. Mostly from [this proyect](https://github.com/Eiji-Kb/simple-pix2pix-pytorch/blob/master/models.py).
	- train.py: Loading of images and training.
	- eval.py: Evaluation of a pretrained model.

- heat_map.ipynb: Notebook to evaluate a pretrained model and visualize regions of maximum change between input and output images.
- pixel_faces.ipynb: Almost the same that can be found in pix2pix_faces but in Notebook.

## Use
### Step 1 - Get training data.

A total number of 183 was downloaded from Google and pixeled with GIMP. Even though the training set is really small the results are quite satisfactory.

Variety in the images is desired. For this reason group photographs, portraits, back turned people (the net should change nothing in this images) were used as training set.

Images are saved in two folders, pixeled and not pixeled, having the images the same name in both of them.

### Step 2 - Training.

The training process is executed:

```bash
python train.py --max_epoch=100000 
                --ROOT=../../Images/ 
		--INPUT=no-pixeled 
		--OUTPUT=pixeled 
		--tr_per=0.8
```

where ```--tr_per``` is the percentage of data in the training set, ```--ROOT``` is the directory having both folders, ```--INPUT``` is the folder name of input images and ```--OUTPUT``` is the folder name with the target images. To pixel faces, ```--INPUT``` is the directory with not pixeled images and ```--OUTPUT``` the directory with the pixeled images.

A pretrained model can be loaded, from which the training continues:

```bash
python train.py --model=checkpoint.pth 
                --max_epoch=100000 
		--ROOT=../../Images/ 
		--INPUT=no-pixeled 
		--OUTPUT=pixeled 
		--tr_per=0.8
```

Every 100 epochs a set of images in printed to check the training process. Also, a file "checkpoint.pth" is saved with the model parameters.


### Step 3 - Results visualization.
To visualize results it is recommended to use "heat_map.ipynb" notebook. Following the instructions in it, we can load a trained model, introduce an image, check the result and plot a heatmap with the regions of maximum change between input and output images:

![Not exit](images/not-exist.png)
[This person does not exist](https://thispersondoesnotexist.com/)

![G20](images/g20.png)
[Imagen Original](https://www.flickr.com/photos/whitehouse/48144069691)

It is possible to download this model from [Drive](https://drive.google.com/open?id=1OF-XhbLZ_YrMYwZJtpjUxiFJ_VBguhOx) for a limited time (depending on my space requierements).
