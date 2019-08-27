# Introduction:

A facial recognition system is a technology capable of identifying or verifying a person from a digital image or a video frame from a video source. At a minimum, a simple real-time facial recognition system is composed of the following pipeline:

Face Enrollment.
Registering faces to a database which includes pre-computing the face embeddings and training a classifier on top of the face embeddings of registered individuals. 
Face Capture.
Reading a frame image from a camera source.
Face Detection.
Detecting faces in a frame image.
Face Encoding/Embedding.
Generating a mathematical representation of each face (coined as embedding) in the frame image.
Face Identification.
Infering each face embedding in an image with face embeddings of known people in a database.

More complex systems include features such as Face Liveness Detection (to counter spoofing attacks via photo, video or 3d mask), face alignment, face augmentation (to increase the number of dataset of images) and face verification (to confirm prediction by comparing cosine similarity or euclidean distance with each database embedding).


# Features:

Having several dataset of images per person is not possible for some use cases of Face Recognition. So finding the appropriate model for that balances accuracy and speed on target hardware platform (CPU, GPU, embedded system) is necessary. The trinity of AI is Data, Algorithms and Compute. libfaceid allows selecting each model/algorithm in the pipeline.

libfaceid library supports several models for each step of the Face Recognition pipeline. Some models are faster while some models are more accurate. You can mix and match the models for your specific use-case, hardware platform and system requirements. 

### Face Detection models for detecting face locations
- [Haar Cascade Classifier via OpenCV](https://github.com/opencv/opencv/blob/master/samples/python/facedetect.py)
- [Histogram of Oriented Gradients (HOG) via DLIB](http://dlib.net/face_detector.py.html)
- [Deep Neural Network via DLIB](http://dlib.net/cnn_face_detector.py.html)
- [Single Shot Detector with ResNet-10 via OpenCV](https://github.com/opencv/opencv/blob/3.4.0/samples/dnn/resnet_ssd_face_python.py)
- [Multi-task Cascaded CNN (MTCNN) via Tensorflow](https://github.com/ipazc/mtcnn/blob/master/tests/test_mtcnn.py)
- [FaceNet MTCNN via Tensorflow](https://github.com/davidsandberg/facenet)

### Face Encoding models for generating face embeddings on detected faces
- [Local Binary Patterns Histograms (LBPH) via OpenCV](https://www.python36.com/face-recognition-using-opencv-part-3/)
- [OpenFace via OpenCV](https://www.pyimagesearch.com/2018/09/24/opencv-face-recognition/)
- [ResNet-34 via DLIB](http://dlib.net/face_recognition.py.html)
- [FaceNet (Inception ResNet v1) via Tensorflow](https://github.com/davidsandberg/facenet)
- [VGG-Face (VGG-16, ResNet-50) via Keras](https://github.com/rcmalli/keras-vggface) - TODO
- [OpenFace via Torch and Lua](https://github.com/cmusatyalab/openface) - TODO

### Classification algorithms for Face Identification using face embeddings
- [Naïve Bayes]
- Linear SVM
- RVF SVM
- Nearest Neighbors
- Decision Tree
- Random Forest
- Neural Net
- Adaboost
- QDA

### Face Liveness Detection models for preventing spoofing attacks
- [Eye Movement](https://www.pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/)
- [Mouth Movement](https://github.com/mauckc/mouth-open)
- [Colorspace Histogram Concatenation](https://github.com/ee09115/spoofing_detection)

- Face Pose estimator models for predicting face landmarks 
- Face Age estimator models for predicting age 
- Face Gender estimator models for predicting gender 
- Face Emotion estimator models for predicting facial expression 

### Installation:

        1. Install Python 3 and Python PIP
           Use Python 3.5.3 for Raspberry Pi 3B+ and Python 3.6.6 for Windows
        2. Install the required Python PIP package dependencies using requirements.txt
           pip install -r requirements.txt

           This will install the following dependencies below:
           opencv-python==3.4.3.18
           opencv-contrib-python==3.4.3.18
           numpy==1.15.4
           imutils==0.5.1
           scipy==1.1.0
           scikit-learn==0.20.0
           mtcnn==0.0.8
           tensorflow==1.8.0
           keras==2.0.8
           h5py==2.8.0
           facenet==1.0.3
           flask==1.0.2
           dlib==19.16.0 # requires CMake
           
           // Installing dlib
           1. Install cmake from https://cmake.org/download/ OR 
           2. pip install https://files.pythonhosted.org/packages/0e/ce/f8a3cff33ac03a8219768f0694c5d703c8e037e6aba2e865f9bae22ed63c/dlib-19.8.1-cp36-cp36m-win_amd64.whl#sha256=794994fa2c54e7776659fddb148363a5556468a6d5d46be8dad311722d54bfcf 


        3. Optional: Install the required Python PIP package dependencies for speech synthesizer and speech recognition for voice capability 
           pip install -r requirements_with_voicecapability.txt

           This will install additional dependencies below:
           playsound==1.2.2
           inflect==0.2.5
           librosa==0.4.2
           unidecode==0.4.20
           pyttsx3==2.7
           gtts==2.0.3
      

           Additional items to install: 
           On Windows, install pypiwin32 using "pip install pypiwin32==223"
           On RPI, 
               sudo apt-get install espeak
               sudo apt-get install python-espeak
               sudo apt-get install portaudio19-dev
               pip3 install pyaudio

### Quickstart (Dummy Guide):

        1. Add your dataset
           ex. datasets/person1/1.jpg, datasets/person2/1.jpg
        2. Train your model with your dataset
           Update training.bat to specify your chosen models
           Run training.bat
        3. Test your model
           Update testing_image.bat to specify your chosen models
           Run testing_image.bat
           
### Pre-requisites:

        1. Add the dataset of images under the datasets directory
           The datasets folder should be in the same location as the test applications.
           Having more images per person makes accuracy much better.
           If only 1 image is possible, then do data augmentation.
             Example:
             datasets/Person1 - contain images of person name Person1
             datasets/Person2 - contain images of person named Person2 
             ...
             datasets/PersonX - contain images of person named PersonX 
        2. Train the model using the datasets. 
           Can use training.py
           Make sure the models used for training is the same for actual testing for better accuracy.


### Examples:

        detector models:           0-HAARCASCADE, 1-DLIBHOG, 2-DLIBCNN, 3-SSDRESNET, 4-MTCNN, 5-FACENET
        encoder models:            0-LBPH, 1-OPENFACE, 2-DLIBRESNET, 3-FACENET
        classifier algorithms:     0-NAIVE_BAYES, 1-LINEAR_SVM, 2-RBF_SVM, 3-NEAREST_NEIGHBORS, 4-DECISION_TREE, 5-RANDOM_FOREST, 6-NEURAL_NET, 7-ADABOOST, 8-QDA
        liveness models:           0-EYESBLINK_MOUTHOPEN, 1-COLORSPACE_YCRCBLUV
        camera resolution:         0-QVGA, 1-VGA, 2-HD, 3-FULLHD

        1. Training with datasets
            Usage: python training.py --detector 0 --encoder 0 --classifier 0
            Usage: python training.py --detector 0 --encoder 0 --classifier 0 --setsynthesizer True --synthesizer 0

        2. Testing with images
            Usage: python testing_image.py --detector 0 --encoder 0 --image datasets/rico/1.jpg

        3. Testing with a webcam
            Usage: python testing_webcam.py --detector 0 --encoder 0 --webcam 0 --resolution 0
            Usage: python testing_webcam_flask.py
                   Then open browser and type http://127.0.0.1:5000 or http://ip_address:5000
                
        4. Testing with a webcam with anti-spoofing attacks
            Usage: python testing_webcam_livenessdetection.py --detector 0 --encoder 0 --liveness 0 --webcam 0 --resolution 0

        5. Testing age/gender/emotion detection
            Usage: python agegenderemotion_webcam.py --detector 0 --webcam 0 --resolution 0
            Usage: python agegenderemotion_webcam_flask.py
                   Then open browser and type http://127.0.0.1:5000 or http://ip_address:5000


### Training models with dataset of images:

        from libfaceid.detector import FaceDetectorModels, FaceDetector
        from libfaceid.encoder  import FaceEncoderModels, FaceEncoder
        from libfaceid.classifier  import FaceClassifierModels

        INPUT_DIR_DATASET         = "datasets"
        INPUT_DIR_MODEL_DETECTION = "models/detection/"
        INPUT_DIR_MODEL_ENCODING  = "models/encoding/"
        INPUT_DIR_MODEL_TRAINING  = "models/training/"

        face_detector = FaceDetector(model=FaceDetectorModels.DEFAULT, path=INPUT_DIR_MODEL_DETECTION)
        face_encoder = FaceEncoder(model=FaceEncoderModels.DEFAULT, path=INPUT_DIR_MODEL_ENCODING, path_training=INPUT_DIR_MODEL_TRAINING, training=True)
        face_encoder.train(face_detector, path_dataset=INPUT_DIR_DATASET, verify=verify, classifier=FaceClassifierModels.NAIVE_BAYES)

### Face Recognition on images:

        import cv2
        from libfaceid.detector import FaceDetectorModels, FaceDetector
        from libfaceid.encoder  import FaceEncoderModels, FaceEncoder

        INPUT_DIR_MODEL_DETECTION = "models/detection/"
        INPUT_DIR_MODEL_ENCODING  = "models/encoding/"
        INPUT_DIR_MODEL_TRAINING  = "models/training/"

        image = cv2.VideoCapture(imagePath)
        face_detector = FaceDetector(model=FaceDetectorModels.DEFAULT, path=INPUT_DIR_MODEL_DETECTION)
        face_encoder = FaceEncoder(model=FaceEncoderModels.DEFAULT, path=INPUT_DIR_MODEL_ENCODING, path_training=INPUT_DIR_MODEL_TRAINING, training=False)

        frame = image.read()
        faces = face_detector.detect(frame)
        for (index, face) in enumerate(faces):
            face_id, confidence = face_encoder.identify(frame, face)
            label_face(frame, face, face_id, confidence)
        cv2.imshow(window_name, frame)
        cv2.waitKey(5000)

        image.release()
        cv2.destroyAllWindows()


### Basic Real-Time Face Recognition (w/a webcam):

        import cv2
        from libfaceid.detector import FaceDetectorModels, FaceDetector
        from libfaceid.encoder  import FaceEncoderModels, FaceEncoder

        INPUT_DIR_MODEL_DETECTION = "models/detection/"
        INPUT_DIR_MODEL_ENCODING  = "models/encoding/"
        INPUT_DIR_MODEL_TRAINING  = "models/training/"

        camera = cv2.VideoCapture(webcam_index)
        face_detector = FaceDetector(model=FaceDetectorModels.DEFAULT, path=INPUT_DIR_MODEL_DETECTION)
        face_encoder = FaceEncoder(model=FaceEncoderModels.DEFAULT, path=INPUT_DIR_MODEL_ENCODING, path_training=INPUT_DIR_MODEL_TRAINING, training=False)

        while True:
            frame = camera.read()
            faces = face_detector.detect(frame)
            for (index, face) in enumerate(faces):
                face_id, confidence = face_encoder.identify(frame, face)
                label_face(frame, face, face_id, confidence)
            cv2.imshow(window_name, frame)
            cv2.waitKey(1)

        camera.release()
        cv2.destroyAllWindows()


### Real-Time Face Recognition With Liveness Detection (w/a webcam):

        import cv2
        from libfaceid.detector import FaceDetectorModels, FaceDetector
        from libfaceid.encoder  import FaceEncoderModels, FaceEncoder
        from libfaceid.liveness import FaceLivenessModels, FaceLiveness

        INPUT_DIR_MODEL_DETECTION  = "models/detection/"
        INPUT_DIR_MODEL_ENCODING   = "models/encoding/"
        INPUT_DIR_MODEL_TRAINING   = "models/training/"
        INPUT_DIR_MODEL_ESTIMATION = "models/estimation/"
        INPUT_DIR_MODEL_LIVENESS   = "models/liveness/"

        camera = cv2.VideoCapture(webcam_index)
        face_detector = FaceDetector(model=FaceDetectorModels.DEFAULT, path=INPUT_DIR_MODEL_DETECTION)
        face_encoder = FaceEncoder(model=FaceEncoderModels.DEFAULT, path=INPUT_DIR_MODEL_ENCODING, path_training=INPUT_DIR_MODEL_TRAINING, training=False)
        face_liveness = FaceLiveness(model=model_liveness, path=INPUT_DIR_MODEL_ESTIMATION)
        face_liveness2 = FaceLiveness(model=FaceLivenessModels.COLORSPACE_YCRCBLUV, path=INPUT_DIR_MODEL_LIVENESS)

        while True:
            frame = camera.read()
            faces = face_detector.detect(frame)
            for (index, face) in enumerate(faces):

                // Check if eyes are close and if mouth is open
                eyes_close, eyes_ratio = face_liveness.is_eyes_close(frame, face)
                mouth_open, mouth_ratio = face_liveness.is_mouth_open(frame, face)

                // Detect if frame is a print attack or replay attack based on colorspace
                is_fake_print  = face_liveness2.is_fake(frame, face)
                is_fake_replay = face_liveness2.is_fake(frame, face, flag=1)

                // Identify face only if it is not fake and eyes are open and mouth is close
                if is_fake_print or is_fake_replay:
                    face_id, confidence = ("Fake", None)
                elif not eyes_close and not mouth_open:
                    face_id, confidence = face_encoder.identify(frame, face)

                label_face(frame, face, face_id, confidence)

            // Monitor eye blinking and mouth opening for liveness detection
            total_eye_blinks, eye_counter = monitor_eye_blinking(eyes_close, eyes_ratio, total_eye_blinks, eye_counter, eye_continuous_close)
            total_mouth_opens, mouth_counter = monitor_mouth_opening(mouth_open, mouth_ratio, total_mouth_opens, mouth_counter, mouth_continuous_open)

            cv2.imshow(window_name, frame)
            cv2.waitKey(1)

        camera.release()
        cv2.destroyAllWindows()



### Real-Time Face Pose/Age/Gender/Emotion Estimation (w/a webcam):

        import cv2
        from libfaceid.detector import FaceDetectorModels, FaceDetector
        from libfaceid.pose import FacePoseEstimatorModels, FacePoseEstimator
        from libfaceid.age import FaceAgeEstimatorModels, FaceAgeEstimator
        from libfaceid.gender import FaceGenderEstimatorModels, FaceGenderEstimator
        from libfaceid.emotion import FaceEmotionEstimatorModels, FaceEmotionEstimator

        INPUT_DIR_MODEL_DETECTION       = "models/detection/"
        INPUT_DIR_MODEL_ENCODING        = "models/encoding/"
        INPUT_DIR_MODEL_TRAINING        = "models/training/"
        INPUT_DIR_MODEL_ESTIMATION      = "models/estimation/"

        camera = cv2.VideoCapture(webcam_index)
        face_detector = FaceDetector(model=FaceDetectorModels.DEFAULT, path=INPUT_DIR_MODEL_DETECTION)
        face_pose_estimator = FacePoseEstimator(model=FacePoseEstimatorModels.DEFAULT, path=INPUT_DIR_MODEL_ESTIMATION)
        face_age_estimator = FaceAgeEstimator(model=FaceAgeEstimatorModels.DEFAULT, path=INPUT_DIR_MODEL_ESTIMATION)
        face_gender_estimator = FaceGenderEstimator(model=FaceGenderEstimatorModels.DEFAULT, path=INPUT_DIR_MODEL_ESTIMATION)
        face_emotion_estimator = FaceEmotionEstimator(model=FaceEmotionEstimatorModels.DEFAULT, path=INPUT_DIR_MODEL_ESTIMATION)

        while True:
            frame = camera.read()
            faces = face_detector.detect(frame)
            for (index, face) in enumerate(faces):
                age = face_age_estimator.estimate(frame, face_image)
                gender = face_gender_estimator.estimate(frame, face_image)
                emotion = face_emotion_estimator.estimate(frame, face_image)
                shape = face_pose_estimator.detect(frame, face)
                face_pose_estimator.add_overlay(frame, shape)
                label_face(age, gender, emotion)
            cv2.imshow(window_name, frame)
            cv2.waitKey(1)

        camera.release()
        cv2.destroyAllWindows()

### Face Enrollment

- Should support dynamic enrollment of faces. Tied up with the maximum number of users the existing system supports.
- Should ask user to move/rotate face (in a circular motion) in order to capture different angles of the face. This gives the system enough flexbility to recognize you at different face angles.
- IPhone X Face ID face enrollment is done twice for some reason. It is possible that the first scan is for liveness detection only.
- How many images should be captured? We can store as much image as possible for better accuracy but memory footprint is the limiting factor. Estimate based on size of 1 picture and the maximum number of users.
- For security purposes and memory related efficiency, images used during enrollment should not be saved. 
Only the mathematical representations (128-dimensional vector) of the face should be used.


### Face Capture

- Camera will be about 1 foot away from user (Apple Face ID: 10-20 inches).
- Camera resolution will depend on display panel size and display resolutions. QVGA size is acceptable for embedded solutions. 
- Take into consideration a bad lighting and extremely dark situation. Should camera have a good flash/LED to emit some light. Iphone X has an infrared light to better perform on dark settings.


### Face Detection

- Only 1 face per frame is detected.
- Face is expected to be within a certain location (inside a fixed box or circular region).
- Detection of faces will be triggered by a user action - clicking some button. (Not automatic detection).
- Face alignment may not be helpful as users can be enforced or directed to have his face inside a fixed box or circular region so face is already expected to be aligned for the most cases. But if adding this feature does not affect speed performance, then face alignment ahould be added if possible.
- Should verify if face is alive via anti-spoofing techniques against picture-based attacks, video-based attacks and 3D mask attacks. Two popular example of liveness detection is detecting eye blinking and mouth opening. 


### Face Encoding/Embedding

- Speed is not a big factor. Face embedding and face identification can take 3-5 seconds.
- Accuracy is critically important. False match rate should be low as much as possible. 
- Can do multiple predictions and get the highest count. Or apply different models for predictions for double checking.


### Face Identification

- Recognize only when eyes are not closed and mouth is not open
- Images per person should at least be 50 images. Increase the number of images per person by cropping images with different face backgound margin, slight rotations, flipping and scaling.
- Classification model should consider the maximum number of users to support. For example, SVM is known to be good for less than 100k classes/persons only.
- Should support unknown identification by setting a threshold on the best prediction. If best prediction is too low, then consider as Unknown.
- Set the number of consecutive failed attempts allowed before disabling face recognition feature. Should fallback to passcode authentication if identification encounters trouble recognizing people.
- Images used for successful scan should be added to the existing dataset images during face enrollment making it adaptive and updated so that a person can be recognized with better accuracy in the future even with natural changes in the face appearance (hairstyle, mustache, pimples, etc.)

In addition to these guidelines, the face recognition solution should provide a way to disable/enable this feature as well as resetting the stored datasets during face enrollment.

### Face Enrollment

- For each person who registers/enrolls to the system, create an audio file "PersonX.wav" for some input text such as "Hello PersonX".
  
### Face Identification

- When a person is identified to be part of the database, we play the corresponding audio file "PersonX.wav". 



# Performance Optimizations:

Speed and accuracy is often a trade-off. Performance can be optimized depending on your specific use-case and system requirements. Some models are optimized for speed while others are optimized for accuracy. Be sure to test all the provided models to determine the appropriate model for your specific use-case, target platform (CPU, GPU or embedded) and specific requirements. Below are additional suggestions to optimize performance.

### Speed
- Reduce the frame size for face detection.
- Perform face recognition every X frames only
- Use threading in reading camera source frames or in processing the camera frames.
- Update the library and configure the parameters directly.

### Accuracy
- Add more datasets if possible (ex. do data augmentation). More images per person will often result to higher accuracy.
- Add face alignment if faces in the datasets are not aligned or when faces may be unaligned in actual deployment.
- Update the library and configure the parameters directly.



# References:

Below are links to valuable resoures. Special thanks to all of these guys for sharing their work on Face Recognition. Without them, learning Face Recognition would be difficult.

### Codes
- [OpenCV tutorials by Adrian Rosebrock](https://www.pyimagesearch.com/2018/09/24/opencv-face-recognition/)
- [Dlib by Davis King](https://github.com/davisking/dlib)
- [Face Recognition (Dlib wrapper) by Adam Geitgey](https://github.com/ageitgey/face_recognition)
- [FaceNet implementation by David Sandberg](https://github.com/davidsandberg/facenet)
- [OpenFace (FaceNet implementation) by Satyanarayanan](https://github.com/cmusatyalab/openface)
- [VGG-Face implementation by Refik Can Malli](https://github.com/rcmalli/keras-vggface)



