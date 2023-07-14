# AI-Generated Image Filter
A tool to filter out unqualified AI-Generated Images
Given a set of reference images, which contains qualified reference images and unqualified reference images (optional), filter out the unqualified AI-Generated Images

## Usage
Put all the qualified reference images in the "qualified_reference_img" folder and all the unqualified images in the "unqualified_reference_img" folder. Put all the AI-Generated images in the "ai_generated" folder.\
Run main.py\
The program would generate folders called "qualified_ai_generated_images_kmeans" and "qualified_ai_generated_images_svm" which contains all the qualified AI-Generated images produced by kmeans and svm methods respectively.\
Note that when there is no unqualified reference images as input, the program would not generate "qualified_ai_generated_images_svm".\
Note that you might want to change the threshold value in kmeans.py for your own dataset.

## Example
If we put a set of images of apple in the "qualified_reference_img" folder and a set of images of oranges in the "unqualified_reference_img" folder, then run 
```
python main.py
```
it would generate two new folders in the same directory and the result is as follows\
Qualified images produced by svm:\
![image](https://github.com/helenzhangyc/AIGeneratedImageFilter/assets/45017130/c1ddd7c2-1cb5-4451-adc0-e7fd80c4d53b)
Qualified images produced by k-means:\
![image](https://github.com/helenzhangyc/AIGeneratedImageFilter/assets/45017130/c8bf3e7c-7843-4327-9474-d7332371b28e)

