# AI-Generated Image Filter
A tool to filter out unqualified AI-Generated Images
Given a set of reference images, which contains qualified reference images and unqualified reference images (optional), filter out the unqualified AI-Generated Images

## Usage
Put all the qualified reference images in the "qualified_reference_img" folder and all the unqualified images in the "unqualified_reference_img" folder. Put all the AI-Generated images in the "ai_generated" folder.\
Run main.py\
The program would generate folders called "qualified_ai_generated_images_kmeans" and "qualified_ai_generated_images_svm" which contains all the qualified AI-Generated images produced by kmeans and svm methods respectively.\
Note that when there is no unqualified reference images as input, the program would not generate "qualified_ai_generated_images_svm".
