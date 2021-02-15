# StyleCLIP

In this repo is the code used to generate face images from prompts in [Generating Images from Prompts using CLIP and StyleGAN](https://towardsdatascience.com/generating-images-from-prompts-using-clip-and-stylegan-1f9ed495ddda).

## Envirionment setup
After cloning this repo, enter into the StyleCLIP folder and run the following command to create a new conda environment named "styleclip" with all the required packages.
```console
conda env create -f environment.yml
```

## Generate faces
Run the following command to generate a face with a custom prompt. In this case the prompt is "The image of a woman with blonde hair and purple eyes"
```console
python clip_generate.py --prompt "The image of a woman with blonde hair and purple eyes"
```

The results will be stored under the folder `generations` with the name of the prompt that you have entered.

