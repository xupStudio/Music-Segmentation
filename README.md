# Music Segmentation Estimation using Self-similarity Matrix 
## Generate SSM
First, transfer the audio into chroma vector, then generate the Self-Similarity Matrix (SSM) of chroma with cosine-similarity.  
The following picture is the SSM of chroma.
![image](https://github.com/CodeGoood/Music-Segmentation/blob/master/pic/ssm%20bigger.png)  
## Find the novelty
Next, we need to distinct the lines between every adjacent blocks. The lines called novelty. In order to find the novelty, we use a simple kernel to scan the SSM from bottom-left to top-right.  
![image](https://github.com/CodeGoood/Music-Segmentation/blob/master/pic/ssm-dir.png)
Then, we plot the output value as the novelty curve. Following picture is the novelty curve.
![image](https://github.com/CodeGoood/Music-Segmentation/blob/master/pic/ssm_to_curve.png)
## Post-process
Finally, we do post-process to the novelty curve with the following rules.  
1. Min distance between peaks: 2 sec  
2. Discard the peaks of low novelty (last 15%)
## Result
https://github.com/CodeGoood/Music-Segmentation/blob/master/pic/output.m4a  
The peak sound in the audio is the phrase position we predict. 
## Challenges
1. A phrase may consist of multiple chords  
2. A phrase may begin/end within a chord  
3. Need information more than chroma (e.g. vocal onset/offset)
