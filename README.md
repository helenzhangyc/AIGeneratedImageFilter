# AI-Generated Image Filter
A tool to filter out unqualified AI-generated images
Given a set of reference images, which contains qualified reference images and unqualified reference images (optional), filter out the unqualified AI-generated images

## Usage
- Put all the qualified reference images in the "qualified_reference_img" folder and all the unqualified reference images in the "unqualified_reference_img" folder. If you do not have any sample unqualified images as references, then you can leave the "unqualified_reference_img" folder empty.
- Put all the AI-generated images in the "ai_generated" folder.\
- Run main.py\
- The program would generate folders called "qualified_ai_generated_images_kmeans" and "qualified_ai_generated_images_svm" which contain all the qualified AI-generated images produced by kmeans and SVM methods respectively.\
- Note that when there are no unqualified reference images as input, the program will not generate "qualified_ai_generated_images_svm" since the SVM method requires some sample unqualified reference images.\
- Note that you might want to change the threshold value and k value in kmeans.py for your own dataset. You might also want to change the kernel function for SVM in svm.py.

## Example
If we put a set of images of apples in the "qualified_reference_img" folder and a set of images of oranges in the "unqualified_reference_img" folder, then run 
```
python main.py
```
It would generate two new folders in the same directory and the result is as follows\
Qualified images produced by svm:\
![image](https://github.com/helenzhangyc/AIGeneratedImageFilter/assets/45017130/c1ddd7c2-1cb5-4451-adc0-e7fd80c4d53b)
Qualified images produced by k-means:\
![image](https://github.com/helenzhangyc/AIGeneratedImageFilter/assets/45017130/c8bf3e7c-7843-4327-9474-d7332371b28e)

