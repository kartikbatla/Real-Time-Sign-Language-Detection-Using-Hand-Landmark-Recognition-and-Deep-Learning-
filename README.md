# Real-Time-Sign-Language-Detection-Using-Hand-Landmark-Recognition-and-Deep-Learning-
This project aims to develop a real-time sign language recognition system using computer vision 
and deep learning techniques to bridge this gap. The system captures hand gestures through a 
webcam, processes them using MediaPipe hand landmarks, and classifies them into predefined 
signs using a trained Artificial Neural Network (ANN) model. 

# Introduction 
American sign language is a predominant sign language Since the only disability D&M people 
have been communication related and they cannot use spoken languages hence the only way for 
them to communicate is through sign language. Communication is the process of exchange of 
thoughts and messages in various ways such as speech, signals, behavior and visuals. Deaf and 
dumb(D&M) people make use of their hands to express different gestures to express their ideas 
with other people. Gestures are the nonverbally exchanged messages and these gestures are 
understood with vision. This nonverbal communication of deaf and dumb people is called sign 
language. 
In our project we basically focus on producing a model which can recognise Fingerspelling based 
hand gestures in order to form a complete word by combining each gesture. The gestures we aim 
to train are as given in the image below.

<img width="678" height="311" alt="image" src="https://github.com/user-attachments/assets/ba9d75fa-12ed-44d5-aacf-127712339d99" />

# Motivation 
The motivation behind this project stems from the need for accessible and affordable 
communication technology for the hearing and speech-impaired community. 
Existing commercial solutions often require expensive hardware or have limited sign vocabularies. 
By developing a low-cost, vision-based, real-time gesture recognition system, this project aims 
to: 
• Empower differently-abled individuals to communicate effectively. 
• Provide a scalable model that can be extended to different regional sign languages. 
• Contribute to research in human-computer interaction and assistive technologies. 
 
# Objectives 
● According to statistics, over 80% of specially abled individuals are illiterate, and the system tries 
to bridge the gap between a normal, a hearing-impaired and a visually impaired person by turning 
a majority of sign language to text and speech.  
● People who are deaf or hard of hearing can communicate their message using gestures that can 
be read.  
● People who are not visually challenged can use the software to comprehend sign language and 
communicate effectively with those who are. Also, people who are visually impaired can also 
communicate when the sign language predicted text is converted to speech.  
● This project will bridge the gap of difficulty in understanding sign language that existed 
previously. 
● To train a model, it will employ cutting-edge deep learning algorithms. The model will collect 
frames for gestures using the camera, train the model, and evaluate the precision for each gesture. 
The gesture will then be predicted in real time. The gesture is then translated to text and speech.  
● The objective of this project is to identify the symbolic expression through images so that the 
communication gap between a normal and hearing-impaired person can be easily bridged by: 
i. Creating data with respect to American sign language &pre- process it.  
ii. Training the pre-processed data with Deep Learning based models to perform sign 
language recognition & speech conversion in real time.  
iii. Testing the model in the real-world scenario. 
 
 
 
 
 
# Literature Review 
Previous researchers have emphasized their work on the prediction of sign language gestures 
to support people with hearing impairments using advanced technologies with artificial 
intelligence algorithms. Although much research has been conducted for SLR, there are still 
limitations and improvements that need to be addressed to improve the hard-of-hearing 
community. This section presents a brief literature review of recent studies on SLR using 
sensor and vision-based deep learning techniques. 
 Literature review of the problem shows that there have been several approaches to address the 
issue of gesture recognition in video using several different methods. In [1] the authors used 
Hidden Markov Models (HMM) to recognize facial expressions from video sequences 
combined with Bayesian Network Classifiers and Gaussian Tree Augmented Naive Bayes 
Classifiers.  
Francois et al. [2] also published a paper on Human Posture Recognition in a Video Sequence 
using methods based on 2D and 3D appearance. The work mentions using PCA to recognize 
silhouettes from a static camera and then using 3D to model posture for recognition. This 
approach has the drawback of having intermediary gestures which may lead to ambiguity in 
training and therefore a lower accuracy in prediction 
This project advances the field by merging lightweight ANN modeling with a custom 
dataset, delivering near state-of-the-art accuracy in a scalable, real-time system. 
 
# Gaps Identified 
1. **Hardware Dependency**: Many existing sign language recognition systems depend on 
specialized hardware such as sensor gloves, Kinect cameras, or depth sensors. This 
increases cost, limits portability, and makes large-scale adoption difficult, especially in 
resource-constrained environments. 
2. **Dataset Limitation**: Most publicly available datasets, like RWTH-PHOENIX or ASL 
Fingerspelling, contain limited gestures or are focused on specific languages such as 
American Sign Language (ASL). There is a lack of diverse, region-specific, and real-world 
gesture datasets covering multiple languages. 
3. **Scalability Issues**: Models trained on specific datasets often fail to generalize across 
different environments, lighting conditions, or hand shapes. This lack of adaptability 
reduces the model’s robustness in real-world applications. 
4. **Real-Time Constraints**: Many high-accuracy models are computationally heavy, 
resulting in high latency and poor real-time performance, especially on low-end hardware 
or edge devices. 
5. **Multilingual Sign Support**: Existing systems primarily focus on English-based sign 
languages and rarely support translation or speech generation in regional or multilingual 
contexts. This restricts accessibility for non-English-speaking users. 
 
# Novelty & Innovation of the Proposal 
• **Efficient Landmark-Based Processing**: The system leverages MediaPipe hand landmarks 
instead of raw image inputs, significantly reducing computational complexity and enabling 
smooth real-time performance on standard hardware. 
• **Custom Dataset Creation**: A self-collected dataset was used to train the model, ensuring 
better personalization, improved accuracy for region-specific gestures, and minimized 
overfitting compared to generic datasets. 
• **High Accuracy Without GPU Dependency**: The model achieves an impressive ~98.98% 
accuracy while being fully capable of running on CPU-only systems, making it accessible 
and cost-effective. 
• **Real-Time Voice Integration**: The system integrates pyttsx3-based real-time speech 
output, providing instant auditory feedback and bridging the communication gap between 
signers and non-signers. 
• **Dynamic and Scalable Design**: The architecture allows for easy gesture expansion — new 
signs can be added seamlessly by collecting additional samples and retraining, ensuring long
term scalability and adaptability. 
 
# Proposed Methodology 
# Background 
**What is MediaPipe?**
Creating pipelines for processing perceptual data including photos, movies, and audio requires 
making the system compliant with the MediaPipe, a hybrid open-source architecture. Real-time 
hand tracking and gesture detection are accomplished using a thorough approach that makes 
advantage of ML. By precisely identifying sign gestures, it provides more hand and 13 finger 
tracking solutions. Using a MediaPipe Holistic pipeline, we were able to extract the landmarks 
from the position of the torso, hands, and face. 

<img width="705" height="468" alt="image" src="https://github.com/user-attachments/assets/97526a7d-41fe-4535-b684-6da8592728a6" />

For our project we have Hand landmark Detection, below is the image that states the points that 
mediapipe marks on Hand. 

<img width="711" height="360" alt="image" src="https://github.com/user-attachments/assets/e0d88aef-1110-43f3-9c24-90df1a1ffd1b" />
<img width="708" height="462" alt="image" src="https://github.com/user-attachments/assets/f94f25e7-5156-45f0-ae1b-76e592b55597" />

# System Overview 
The proposed system is designed to recognize hand gestures in real-time and translate them into 
text and speech, thereby improving accessibility for individuals with hearing or speech 
impairments. It functions through the following key stages: 
**1. Data Collection:** The system captures live video frames using a webcam. Each frame is 
processed through MediaPipe Hand Tracking, which extracts 21 key landmarks per 
hand. These landmarks represent critical finger joints and hand positions, serving as the 
primary input features instead of raw pixel data. 
**2. Preprocessing:** The extracted landmark coordinates are normalized to maintain 
consistency across different distances, lighting conditions, and hand sizes. Each gesture is 
labeled accordingly (e.g., A–Z, numbers, or custom signs) and stored in a structured dataset 
for model training. 

<img width="711" height="614" alt="image" src="https://github.com/user-attachments/assets/ed9fe8fa-6f57-4a9f-8310-4bd0308517fd" />

**3. Model Training:** A Feedforward Artificial Neural Network (ANN) is trained using the 
preprocessed landmark data. The model learns to map specific spatial landmark patterns to 
their corresponding gesture classes. The training is performed on a custom self-created 
dataset, ensuring high accuracy and adaptability to user-specific gestures. 

<img width="711" height="328" alt="image" src="https://github.com/user-attachments/assets/78efcfb3-57fd-44ac-9c3a-463369642f33" />

**4. Real-Time Inference:** During live operation, the webcam continuously streams frames to 
the system. The MediaPipe model detects and extracts landmarks in real time, which are 
then passed through the trained ANN to predict the gesture label with high accuracy.
**6. Voice Output:** The predicted gesture label is instantly converted into spoken words using 
the pyttsx3 text-to-speech engine. This feature enables effective communication with 
non-signers by producing audible translations of the detected signs.

<img width="583" height="364" alt="image" src="https://github.com/user-attachments/assets/4d181486-b664-47bf-99ce-1dba36919c72" />
