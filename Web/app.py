from PIL.Image import fromarray
from flask import Flask, render_template, request, make_response
from keras.preprocessing.image import load_img
import numpy as np
from PIL import Image
import cv2
from scipy.ndimage import gaussian_filter
from sklearn.utils import resample
from keras.models import load_model
from keras.preprocessing.image import img_to_array

app = Flask(__name__)

# Multi disease detection model
model_multi_disease = load_model('C:/Users/laksh/OneDrive/Desktop/Web/models/vgg19_multi_disease.h5')

# Tumor Detection Models
tumor_vgg_16 = load_model('C:/Users/laksh/OneDrive/Desktop/Web/models/TumorDetectionModel_VGG-16 (1).h5')
tumor_vgg_19 = load_model('C:/Users/laksh/OneDrive/Desktop/Web/models/TumorDetectionModel_VGG-19.h5')
tumor_resnet_50 = load_model('C:/Users/laksh/OneDrive/Desktop/Web/models/TumorDetectionModel_ResNet50.h5')

# Tumour Classification Models
classification_vgg_16 = load_model('C:/Users/laksh/OneDrive/Desktop/Web/models/brain_tumor_classification_vgg16.h5')
classification_vgg_19 = load_model('C:/Users/laksh/OneDrive/Desktop/Web/models/brain_tumor_classification_vgg19.h5')
classification_resnet_50 = load_model(
    'C:/Users/laksh/OneDrive/Desktop/Web/models/brain_tumor_classification_resnet50.h5')

# Side detection Models
model_side_detection = load_model('C:/Users/laksh/OneDrive/Desktop/Web/models/brain_side_vgg16.h5')

# Stroke Detection Models
model_vgg16_stroke = load_model('C:/Users/laksh/OneDrive/Desktop/Web/models/ischemic_stroke_vgg16.h5')
model_resnet50_stroke = load_model('C:/Users/laksh/OneDrive/Desktop/Web/models/ischemic_stroke_resnet50.h5')
model_vgg19_stroke = load_model('C:/Users/laksh/OneDrive/Desktop/Web/models/ischemic_stroke_vgg19.h5')

# Alzheimer Disease Detection Model
model_efficient_net_alzheimer = load_model(
    'C:/Users/laksh/OneDrive/Desktop/Web/models/alzhimer_classification_efficientNet.h5')

# Brain Image Detection Model
model_brain_image_detection = load_model('C:/Users/laksh/OneDrive/Desktop/Web/models/brain_image_detection.h5')


# Load the initial page (login page)
@app.route('/')
def dashboard():
    return render_template('login.html', data="dashboard")


# method to go to the Brain Stroke Detector page
@app.route('/BrainStrokeDetector')
def BrainStrokeDetector():
    return render_template('BrainStrokeDetector.html')


# method to go to the Dashboard page
@app.route('/Dashboard')
def Dashboard():
    return render_template('Dashboard.html')


# method to go to the Brain Tumour Detector page
@app.route('/BrainTumourDetector')
def BrainTumourDetector():
    return render_template('BrainTumourDetector.html')


# method to go to the Alzheimer Disease Detector page
@app.route('/AlzheimerDiseaseDetector')
def AlzheimerDiseaseDetector():
    return render_template('AlzheimerDiseaseDetector.html')


# method to go to the report generator page
@app.route('/ReportGenerator')
def ReportGenerator():
    return render_template('ReportGenerator.html')


# method to go to the register page
@app.route('/Register')
def Register():
    return render_template('Register.html')


# method to go to the login page
@app.route('/login')
def login():
    return render_template('login.html')


# method to go to the login page
@app.route('/account')
def account():
    return render_template('account.html')


# method to go to the forgot password page
@app.route('/forgot')
def forgot():
    return render_template('forgotPassword.html')


# method used to predict tumours
@app.route('/tumor', methods=['POST'])
def predict_tumour_type():
    # Assigning the voting for tumor detection
    label_mapping_detector = {1: "Tumor", 0: "Normal"}

    label_mapping_brain = {1: 'BrainImages', 0: 'NotBrainImage'}

    label_mapping_classification = {0: 'Glioma/Metastasis', 1: 'Meningioma/Metastasis', 3: 'Pituitary',
                                    2: 'NoTumor'}

    # get the image file
    imagefile = request.files['imagefile']
    image_path = "./static/predictingBrainClassificationImages/" + imagefile.filename
    imagefile.save(image_path)

    # Load the image for classification
    classification_image = load_img(image_path, target_size=(256, 256))

    # load the image for detection
    image = load_img(image_path, target_size=(256, 256))

    # convert the image into a numpy array
    image = np.array(image)

    # apply gray scale with gaussian blur
    gray_scale_image = apply_gaussian_gray_scale_filter(image)

    # Prepare the grayscale image for prediction
    brain_image = np.expand_dims(gray_scale_image, axis=0)

    # Predict the class probabilities for the brain image
    class_probabilities = model_brain_image_detection.predict(brain_image)

    # Get the predicted class index
    predicted_class_index = np.argmax(class_probabilities)

    # Get the corresponding class name from the label mapping
    predicted_class_name = label_mapping_brain[predicted_class_index]

    # Get the score for the predicted class
    brain_detection_score = class_probabilities[0][predicted_class_index]

    detector_image = load_img(image_path, target_size=(256, 256))

    # Apply gamma correction to the classification image
    classification_image = apply_gamma_correction(classification_image, 1.5)

    # Apply gamma correction to the detector image
    detector_image = apply_gamma_correction(detector_image, 1.5)

    # Convert PIL Classification image to array
    classification_image_array = img_to_array(classification_image)

    # Convert PIL Detector image to array
    detector_image_array = img_to_array(detector_image)

    # Expand dimensions to match the input shape expected by the model
    classification_image_array = np.expand_dims(classification_image_array, axis=0)

    # Expand dimensions to match the input shape expected by the model
    detector_image_array = np.expand_dims(detector_image_array, axis=0)

    prediction_array = [0, 0]
    score_array = [0, 0]

    # check if the detected image is not a brain image and give the output
    if predicted_class_name == 'NotBrainImage':
        prediction_array[0] = "Not Brain Image"
        prediction_array[1] = "Not Brain Image"
        score_array[0] = "{:.2f}".format(calculate_threshold_probability(brain_detection_score))
        score_array[1] = "{:.2f}".format(calculate_threshold_probability(brain_detection_score))
        return render_template('BrainTumourDetector.html', image_path=image_path, predicted_class=prediction_array,
                               score=score_array)

    # If the given image is a brain image, check the high probability disease
    all_disease_vgg_19_probability = model_multi_disease.predict(classification_image_array)[0]
    all_disease_score = all_disease_vgg_19_probability[np.argmax(all_disease_vgg_19_probability)]
    all_disease_prediction = np.argmax(all_disease_vgg_19_probability)

    # get the output of high probability disease image and see whether there is a tumour
    if all_disease_prediction != 0:
        prediction_array[0] = "Normal"
        prediction_array[1] = "No Tumour"
        score_array[0] = "{:.2f}".format(calculate_threshold_probability(all_disease_score))
        score_array[1] = "{:.2f}".format(calculate_threshold_probability(all_disease_score))
        return render_template('BrainTumourDetector.html', image_path=image_path, predicted_class=prediction_array,
                               score=score_array)

    # get the probabilities of the vgg16 image
    detector_vgg_16_probability = tumor_vgg_16.predict(detector_image_array)[0]
    detector_vgg_19_probability = tumor_vgg_19.predict(detector_image_array)[0]
    detector_resnet50_probability = tumor_resnet_50.predict(detector_image_array)[0]

    # get the highest probability class probability value of the vgg16 model
    detector_score = detector_vgg_16_probability[np.argmax((detector_vgg_16_probability + detector_vgg_19_probability + detector_resnet50_probability) / 3)]

    detector_prediction = np.argmax((detector_vgg_16_probability + detector_vgg_19_probability + detector_resnet50_probability) / 3)

    # get the class of the highest probability value
    detector_class = label_mapping_detector[detector_prediction]

    # assign the highest probability category
    prediction_array[0] = detector_class
    score_array[0] = "{:.2f}".format(calculate_threshold_probability(detector_score))

    # if the tumor detected find the tumour category
    if detector_class == "Tumor":
        probabilities_side = model_side_detection.predict(classification_image_array)[0]

        print(probabilities_side)

        # Predict class probabilities
        probability_vgg16 = classification_vgg_16.predict(classification_image_array)[0]
        probability_vgg19 = classification_vgg_19.predict(classification_image_array)[0]
        probability_resnet50 = classification_resnet_50.predict(classification_image_array)[0]

        # get the probabilities of the vgg16 model
        probabilities = ((probability_vgg16 + probability_vgg19 + probability_resnet50) / 3)

        # get the highest probability class probability
        score = probabilities[np.argmax(probabilities)]

        # Assign the highest probability class
        score_array[1] = "{:.2f}".format(calculate_threshold_probability(score))

        print("Probability : ", score_array[1])

        # Get the predicted class index
        predicted_class_index = np.argmax(probabilities)

        print("Predicted Class Index: ", predicted_class_index)

        # Get the score of the predicted class
        score1 = probabilities[predicted_class_index]

        # check whether a no tumour is detected from the classification model
        if predicted_class_index == 2:
            predicted_class_index = np.partition(probabilities, -2)[-2]
            print("Predicted Class Index: ", predicted_class_index)
            score1 = probabilities[int(predicted_class_index)]
            predicted_class = label_mapping_classification[int(predicted_class_index)]
            prediction_array[1] = predicted_class
        else:
            # Get the predicted class label
            predicted_class = label_mapping_classification[predicted_class_index]
            prediction_array[1] = predicted_class

        print(f"Predicted Class: {predicted_class}, Score: {score1}")

    else:
        # if the tumour is not detected, assign the normal class (No Tumor)
        prediction_array[1] = "No Tumour"
        score_array[1] = "{:.2f}".format(calculate_threshold_probability(detector_score))

    print("Predicted class array:", prediction_array)

    # return the result values to the frontend
    return render_template('BrainTumourDetector.html', image_path=image_path, predicted_class=prediction_array,
                           score=score_array)


# Function to apply gamma correction
def apply_gamma_correction(image, gamma=1.5):
    image_array = np.array(image)

    # Normalize pixel values to the range [0, 1]
    normalized_image = image_array / 255.0

    # Apply gamma correction
    corrected_image = np.power(normalized_image, gamma)

    # Denormalize the image to the original range [0, 255]
    corrected_image = (corrected_image * 255).astype(np.uint8)

    # Convert numpy array back to image
    corrected_image = Image.fromarray(corrected_image)

    return corrected_image


# Apply sobel 8 filter for stroke image prediction
def apply_sobel8_filter(image):
    image = np.array(image)

    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

    # Applying Sobel filter
    sobel_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=-1)
    sobel_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=-1)
    edges = cv2.magnitude(sobel_x, sobel_y)

    # Normalize edges
    edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX)

    # Convert to uint8
    edges = edges.astype('uint8')

    return edges


# Method to predict stroke
@app.route('/stroke', methods=['POST'])
def predict_stroke():
    # Label mapping for side detection
    label_mapping_side = {0: 'Axial', 1: 'Coronal', 3: 'Sagittal'}
    label_mapping_brain = {1: 'BrainImages', 0: 'NotBrainImage'}

    # get the path of the image
    imagefile = request.files['imagefile']
    image_path = "./static/predictingStrokeImages/" + imagefile.filename
    imagefile.save(image_path)

    # load the image from the folder
    image = load_img(image_path, target_size=(256, 256))

    # convert the image into an array
    image = np.array(image)

    # apply gray scale and gaussian blur to reduce the noise in the image
    gray_scale_image = apply_gaussian_gray_scale_filter(image)

    print(gray_scale_image.shape)

    # Prepare the grayscale image for prediction
    brain_image = np.expand_dims(gray_scale_image, axis=0)

    # Predict the class probabilities for the brain image
    class_probabilities = model_brain_image_detection.predict(brain_image)

    # Get the predicted class index
    predicted_class_index = np.argmax(class_probabilities)

    # Get the corresponding class name from the label mapping
    predicted_class_name = label_mapping_brain[predicted_class_index]

    # Get the score for the predicted class
    brain_detection_score = class_probabilities[0][predicted_class_index]

    print(predicted_class_name)

    # check if the detected image is not a brain image and give the output
    if predicted_class_name == 'NotBrainImage':
        class_name = ["Not Brain Image", "Not Brain Image"]
        prediction_score = ["{:.2f}".format(calculate_threshold_probability(brain_detection_score)),
                            "{:.2f}".format(calculate_threshold_probability(brain_detection_score))]
        return render_template('BrainStrokeDetector.html', image_path=image_path, class_name=class_name,
                               prediction_score=prediction_score)

    # Apply sobel 8 filter to the image
    image = apply_sobel8_filter(image)

    # load the image for classification
    classification_image = load_img(image_path, target_size=(256, 256))

    # apply gamma filter (increased to 1.5)
    multi_image_array = apply_gamma_correction(classification_image, 1.5)

    # Convert PIL Classification image to array
    multi_image_array = img_to_array(multi_image_array)

    # Expand dimensions to match the input shape expected by the model
    multi_image_array = np.expand_dims(multi_image_array, axis=0)

    # check the probability of all the diseases in vgg 19
    all_disease_vgg_19_probability = model_multi_disease.predict(multi_image_array)[0]
    all_disease_score = all_disease_vgg_19_probability[np.argmax(all_disease_vgg_19_probability)]
    all_disease_prediction = np.argmax(all_disease_vgg_19_probability)

    # check if the image is not a stroke image
    if all_disease_prediction != 2:
        # get the side prediction of the image
        side_prediction = model_side_detection.predict(multi_image_array)
        side_prediction_score = side_prediction[0][np.argmax(side_prediction)]
        side_class = np.argmax(side_prediction)
        side_name = label_mapping_side[side_class]

        # Check whether the image is normal or not
        class_name = ["Normal", side_name]
        prediction_score = ["{:.2f}".format(calculate_threshold_probability(all_disease_score)),
                            "{:.2f}".format(calculate_threshold_probability(side_prediction_score))]
        return render_template('BrainStrokeDetector.html', image_path=image_path, class_name=class_name,
                               prediction_score=prediction_score)

    # get the side of the image given
    side_prediction = model_side_detection.predict(multi_image_array)
    side_prediction_score = side_prediction[0][np.argmax(side_prediction)]
    side_class = np.argmax(side_prediction)
    side_name = label_mapping_side[side_class]

    print(f"Predicted Side: {side_name}, Score: {side_prediction_score}")

    # label mapping for stroke images
    label_mapping = {0: 'Ischemic', 1: 'Normal'}

    # convert the image into an array
    image = np.expand_dims(image, axis=0)

    vgg_16_prediction = model_vgg16_stroke.predict(image)
    vgg_19_prediction = model_vgg19_stroke.predict(image)
    resnet50_prediction = model_resnet50_stroke.predict(image)

    # predict the class probabilities
    predictions = (vgg_16_prediction + vgg_19_prediction + resnet50_prediction) / 3
    class_name = np.argmax(predictions)

    # Get the prediction score
    prediction_score = predictions[0][class_name]

    # get the class name of the predicted image
    class_name = [label_mapping[class_name], side_name]
    prediction_score = ["{:.2f}".format(calculate_threshold_probability(prediction_score)),
                        "{:.2f}".format(calculate_threshold_probability(side_prediction_score))]

    # return the results to the frontend of the web
    return render_template('BrainStrokeDetector.html', image_path=image_path, class_name=class_name,
                           prediction_score=prediction_score)


# Apply the random up sampler and gaussian filter
def apply_random_up_sampler_gaussian_filter(image):
    # Resample the image to a higher resolution
    sampled_img = resample([image], n_samples=2)[0]
    # Apply Gaussian filter to the resampled image
    filtered_img = gaussian_filter(sampled_img, sigma=1)
    # return the filtered image
    return filtered_img


# apply the gaussian gray scale filter
def apply_gaussian_gray_scale_filter(image):
    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

    # Convert image to grayscale
    gray_scale_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)

    # return gary scale image
    return gray_scale_image


# Method that used to predict the alzheimer disease images
@app.route('/alzheimer', methods=['POST'])
def predict_alzheimer():
    label_mapping_side = {0: 'Axial', 1: 'Coronal', 3: 'Sagittal'}
    label_mapping_brain = {1: 'BrainImages', 0: 'NotBrainImage'}

    # path of the given image
    imagefile = request.files['imagefile']
    image_path = "./static/PredictingAlzheimerImages/" + imagefile.filename

    imagefile.save(image_path)

    # load the image from the image path
    image = load_img(image_path, target_size=(256, 256))

    # Apply the random up sampler and apply gaussian filter
    image = apply_random_up_sampler_gaussian_filter(image)

    # load the image to the system for disease identification
    classification_image = load_img(image_path, target_size=(256, 256))

    # Apply the gray scale filter to the image
    gray_scale_image = apply_gaussian_gray_scale_filter(image)

    # Prepare the grayscale image for prediction
    brain_image = np.expand_dims(gray_scale_image, axis=0)

    # Predict the class probabilities for the brain image
    class_probabilities = model_brain_image_detection.predict(brain_image)

    # Get the predicted class index
    predicted_class_index = np.argmax(class_probabilities)

    # Get the corresponding class name from the label mapping
    predicted_class_name = label_mapping_brain[predicted_class_index]

    # Get the score for the predicted class
    brain_detection_score = class_probabilities[0][predicted_class_index]

    multi_image_array = apply_gamma_correction(classification_image, 1.5)

    # Convert PIL Classification image to array
    multi_image_array = img_to_array(multi_image_array)

    # Expand dimensions to match the input shape expected by the model
    multi_image_array = np.expand_dims(multi_image_array, axis=0)

    # check if the detected image is not a brain image and give the output
    if predicted_class_name == 'NotBrainImage':
        class_name = ["Not Brain Image", "Not Brain Image"]
        prediction_score = ["{:.2f}".format(calculate_threshold_probability(brain_detection_score)),
                            "{:.2f}".format(calculate_threshold_probability(brain_detection_score))]
        return render_template('AlzheimerDiseaseDetector.html', image_path=image_path,
                               predicted_class=class_name,
                               score=prediction_score)

    # get the probabilities of the vgg19 model
    all_disease_vgg_19_probability = model_multi_disease.predict(multi_image_array)[0]
    all_disease_score = all_disease_vgg_19_probability[np.argmax(all_disease_vgg_19_probability)]
    all_disease_prediction = np.argmax(all_disease_vgg_19_probability)

    # check if the image is not a stroke image
    if all_disease_prediction != 1:
        # get the side predictions
        side_prediction = model_side_detection.predict(multi_image_array)
        side_prediction_score = side_prediction[0][np.argmax(side_prediction)]
        side_class = np.argmax(side_prediction)
        side_name = label_mapping_side[side_class]

        # Assign the predicted class names
        class_name = ["Normal", side_name]
        prediction_score = ["{:.2f}".format(calculate_threshold_probability(all_disease_score)),
                            "{:.2f}".format(calculate_threshold_probability(side_prediction_score))]
        # return the results to the frontend
        return render_template('AlzheimerDiseaseDetector.html', image_path=image_path,
                               predicted_class=class_name,
                               score=prediction_score)
    # Label mapping used to label the alzheimer disease
    label_mapping = {'VeryMildDemented': 0, 'NonDemented': 1, 'ModerateDemented': 2, 'MildDemented': 3}

    # Convert PIL image to array
    image_array = img_to_array(image)
    # Expand dimensions to match the input shape expected by the model
    image_array = np.expand_dims(image_array, axis=0)

    # Predict class probabilities
    probabilities = model_efficient_net_alzheimer.predict(image_array)[0]

    # Get the predicted class index
    predicted_class_index = np.argmax(probabilities)

    # Get the predicted class label
    predicted_class = list(label_mapping.keys())[predicted_class_index]

    # Get the score of the predicted class
    score = probabilities[predicted_class_index]

    # get the side prediction results for the image
    side_prediction = model_side_detection.predict(multi_image_array)
    side_prediction_score = side_prediction[0][np.argmax(side_prediction)]
    side_class = np.argmax(side_prediction)
    side_name = label_mapping_side[side_class]

    # assign the predicted results to the output variables
    predicted_class_array = [predicted_class, side_name]

    # assign the score to the output variables
    score_array = ["{:.2f}".format(calculate_threshold_probability(score)),
                   "{:.2f}".format(calculate_threshold_probability(side_prediction_score))]

    # return the output of the image to its frontend
    return render_template('AlzheimerDiseaseDetector.html', image_path=image_path,
                           predicted_class=predicted_class_array,
                           score=score_array)


# disease status dictionary
disease_status = {"Tumour": "Not Detected", "Tumour Type": "Not Detected", "Alzheimer": "Not Detected",
                  "Stroke": "Not Detected", "Edge": "Not Detected"}

# disease score dictionary
disease_score = {"Tumour": 0.00, "Tumour Type": 0.00, "Alzheimer": 0.00, "Stroke": 0.00, "Edge": 0.00}

# get the required image
required_image = None


# Method that is used to generate the report with all disease results
@app.route('/generateReport', methods=['POST'])
def generateReport():
    global disease_status, disease_score
    # dictionaries used to store the image labels
    disease_status = {"Tumour": "Not Detected", "Tumour Type": "Not Detected", "Alzheimer": "Not Detected",
                      "Stroke": "Not Detected", "Edge": "Not Detected"}

    disease_score = {"Tumour": 0.00, "Tumour Type": 0.00, "Alzheimer": 0.00, "Stroke": 0.00, "Edge": 0.00}

    label_mapping_brain = {1: 'BrainImages', 0: 'NotBrainImage'}
    label_mapping_side = {0: 'Axial', 1: 'Coronal', 3: 'Sagittal'}
    label_mapping_detector = {1: "Tumor", 0: "Normal"}
    label_mapping_classification = {0: 'Glioma/Metastasis', 1: 'Meningioma/Metastasis', 3: 'Pituitary',
                                    2: 'NoTumor'}
    label_mapping_alzheimer = {'VeryMildDemented': 0, 'NonDemented': 1, 'ModerateDemented': 2, 'MildDemented': 3}

    # path of the given image
    imagefile = request.files['imagefile']
    image_path = "./static/PredictingAlzheimerImages/" + imagefile.filename

    global required_image
    required_image = image_path

    imagefile.save(image_path)

    # load the required image to the system
    image = load_img(image_path, target_size=(256, 256))

    # convert image to a numpy array
    image = img_to_array(image)

    # apply the gray scaler to the image
    gray_scale_image = apply_gaussian_gray_scale_filter(image)

    # Prepare the grayscale image for prediction
    brain_image = np.expand_dims(gray_scale_image, axis=0)

    # Predict the class probabilities for the brain image
    class_probabilities = model_brain_image_detection.predict(brain_image)

    # Get the predicted class index
    predicted_class_index = np.argmax(class_probabilities)

    # Get the corresponding class name from the label mapping
    predicted_class_name = label_mapping_brain[predicted_class_index]

    # Get the score for the predicted class
    brain_detection_score = class_probabilities[0][predicted_class_index]

    # if the brain image is not detected, display brain not detected massages
    if predicted_class_name == 'NotBrainImage':
        disease_status["Tumour"] = "Not Brain Image"
        disease_score["Tumour"] = "{:.2f}".format(calculate_threshold_probability(brain_detection_score))
        disease_status["Tumour Type"] = "Not Brain Image"
        disease_score["Tumour Type"] = "{:.2f}".format(calculate_threshold_probability(brain_detection_score))
        disease_status["Alzheimer"] = "Not Brain Image"
        disease_score["Alzheimer"] = "{:.2f}".format(calculate_threshold_probability(brain_detection_score))
        disease_status["Stroke"] = "Not Brain Image"
        disease_score["Stroke"] = "{:.2f}".format(calculate_threshold_probability(brain_detection_score))
        disease_status["Edge"] = "Not Brain Image"
        disease_score["Edge"] = "{:.2f}".format(calculate_threshold_probability(brain_detection_score))
        return render_template('ReportGenerator.html', image_path=image_path, score=disease_score,
                               predicted_class_names=disease_status)

    # Apply the gamma filter to reduce the noise
    gamma_image = apply_gamma_correction(image, 1.5)

    # Apply sobel 8 filter for edge detection of the image
    sobel_image = apply_sobel8_filter(gamma_image)

    # Apply gamma filter to the image
    gamma_image_expand = np.expand_dims(gamma_image, axis=0)

    # check the probability of all the diseases in vgg 19 (High probability disease)
    all_disease_vgg_19_probability = model_multi_disease.predict(gamma_image_expand)[0]
    all_disease_score = all_disease_vgg_19_probability[np.argmax(all_disease_vgg_19_probability)]

    # get the maximum class from the output
    all_disease_prediction = np.argmax(all_disease_vgg_19_probability)

    # get the side prediction results of the given image
    side_prediction = model_side_detection.predict(gamma_image_expand)
    side_prediction_score = side_prediction[0][np.argmax(side_prediction)]
    side_class = np.argmax(side_prediction)
    side_name = label_mapping_side[side_class]

    # assign the detected age to the output variables
    disease_status["Edge"] = side_name
    disease_score["Edge"] = "{:.2f}".format(calculate_threshold_probability(side_prediction_score))

    # check if the image is not a stroke image (for display probability)
    if all_disease_vgg_19_probability[2] > 0.5:
        disease_score["Stroke"] = "{:.2f}".format(
            1 - calculate_threshold_probability(all_disease_vgg_19_probability[2]))
    else:
        disease_score["Stroke"] = "{:.2f}".format(
            calculate_threshold_probability(all_disease_vgg_19_probability[2]))

    # check if the image is not a alzheimer image (for display probability)
    if all_disease_vgg_19_probability[1] > 0.5:
        disease_score["Alzheimer"] = "{:.2f}".format(
            1 - calculate_threshold_probability(all_disease_vgg_19_probability[1]))
    else:
        disease_score["Alzheimer"] = "{:.2f}".format(
            calculate_threshold_probability(all_disease_vgg_19_probability[1]))

    # check if the image is not a tumour image (for display probability)
    if all_disease_vgg_19_probability[0] > 0.5:
        disease_score["Tumour"] = "{:.2f}".format(
            calculate_threshold_probability(1 - all_disease_vgg_19_probability[0]))
        disease_score["Tumour Type"] = "{:.2f}".format(
            calculate_threshold_probability(1 - all_disease_vgg_19_probability[0]))
    else:
        disease_score["Tumour"] = "{:.2f}".format(calculate_threshold_probability(all_disease_vgg_19_probability[0]))
        disease_score["Tumour Type"] = "{:.2f}".format(
            calculate_threshold_probability(all_disease_vgg_19_probability[0]))

    # check if the image is a tumour image
    if all_disease_prediction == 0:
        # Apply gamma correction to the classification image
        detector_vgg_16_probability = tumor_vgg_16.predict(gamma_image_expand)[0]
        detector_vgg_19_probability = tumor_vgg_19.predict(gamma_image_expand)[0]
        detector_resnet50_probability = tumor_resnet_50.predict(gamma_image_expand)[0]
        # Get the highest probability class probability
        detector_score = detector_vgg_16_probability[np.argmax(detector_vgg_16_probability)]
        detector_score_19 = detector_vgg_19_probability[np.argmax(detector_vgg_19_probability)]
        detector_score_50 = detector_resnet50_probability[np.argmax(detector_resnet50_probability)]
        # Get the highest probability class
        detector_prediction = np.argmax((detector_vgg_16_probability + detector_score_19 + detector_score_50) / 3)

        # Get the detected class
        if detector_prediction == 1:
            # Predict class probabilities
            probability_vgg16 = classification_vgg_16.predict(gamma_image_expand)[0]
            probability_vgg19 = classification_vgg_19.predict(gamma_image_expand)[0]
            probability_resnet50 = classification_resnet_50.predict(gamma_image_expand)[0]

            # get the probabilities of the vgg16 model with average
            probabilities = ((probability_vgg16 + probability_vgg19 + probability_resnet50) / 3)

            # Get the predicted class index
            predicted_class_index = np.argmax(probabilities)

            # Get the predicted class label
            if predicted_class_index == 2:
                predicted_class_index = int(np.partition(probabilities, -2)[-2])

            # Get the predicted class label
            predicted_class = label_mapping_classification[predicted_class_index]

            # Set the score of the predicted class
            disease_status["Tumour"] = label_mapping_detector[detector_prediction]
            disease_score["Tumour"] = "{:.2f}".format(calculate_threshold_probability(detector_score))

            # Set the predicted class and score
            disease_status["Tumour Type"] = predicted_class
            disease_score["Tumour Type"] = "{:.2f}".format(calculate_threshold_probability(detector_score))

            # return the output variables to the frontend of the web application
            return render_template('ReportGenerator.html', image_path=image_path, score=disease_score,
                                   predicted_class=disease_status)

        else:
            # Set the predicted class (Tumor) and score
            disease_status["Tumour"] = label_mapping_detector[detector_prediction]
            disease_score["Tumour"] = "{:.2f}".format(calculate_threshold_probability(detector_score))

            # Set the predicted class (Tumor Type) and score
            disease_status["Tumour Type"] = label_mapping_detector[detector_prediction]
            disease_score["Tumour Type"] = "{:.2f}".format(calculate_threshold_probability(detector_score))

            # return the output variables to the frontend of the web application
            return render_template('ReportGenerator.html', image_path=image_path, score=disease_score,
                                   predicted_class=disease_status)
    # check if the image is a alzheimer image
    elif all_disease_prediction == 1:
        # load the image for classification
        image = load_img(image_path, target_size=(256, 256))

        # Apply the random up sampler and apply gaussian filter
        image = apply_random_up_sampler_gaussian_filter(image)

        # Convert PIL image to array
        image_array = img_to_array(image)
        # Expand dimensions to match the input shape expected by the model
        image_array = np.expand_dims(image_array, axis=0)

        # Predict class probabilities
        probabilities = model_efficient_net_alzheimer.predict(image_array)[0]

        # Get the predicted class index
        predicted_class_index = np.argmax(probabilities)

        # Get the predicted class label
        predicted_class = list(label_mapping_alzheimer.keys())[predicted_class_index]

        # Get the score of the predicted class
        score = probabilities[predicted_class_index]

        # set the predicted class and score
        disease_status["Alzheimer"] = predicted_class
        disease_score["Alzheimer"] = "{:.2f}".format(calculate_threshold_probability(score))

        # return the output variables to the frontend of the web application
        return render_template('ReportGenerator.html', image_path=image_path, score=disease_score,
                               predicted_class=disease_status)

    # check if the image is a stroke image
    elif all_disease_prediction == 2:
        # Apply gamma filter to the image
        label_mapping = {0: 'Ischemic', 1: 'Not Detected'}

        # convert the image into an array
        image = np.expand_dims(sobel_image, axis=0)

        vgg_16_prediction = model_vgg16_stroke.predict(image)
        vgg_19_prediction = model_vgg19_stroke.predict(image)
        resnet50_prediction = model_resnet50_stroke.predict(image)

        # predict the class probabilities
        predictions = (vgg_16_prediction + vgg_19_prediction + resnet50_prediction) / 3
        class_name = np.argmax(predictions)

        # Get the prediction score
        prediction_score = predictions[0][class_name]

        # set the predicted class and score for stroke
        disease_status["Stroke"] = label_mapping[class_name]

        # set the predicted class and score stroke
        disease_score["Stroke"] = "{:.2f}".format(calculate_threshold_probability(prediction_score))

        # return the output variables to the frontend of the web application
        return render_template('ReportGenerator.html', image_path=image_path, score=disease_score,
                               predicted_class=disease_status)

    # return the output variables to the frontend of the web application
    return render_template('ReportGenerator.html', image_path=image_path)


# Method to generate the report
@app.route('/report')
def report():
    # return the output variables to the frontend of the web application
    return render_template('report.html', score=disease_score,
                           predicted_class=disease_status, image_path=required_image)


# Method to calculate the threshold probability
def calculate_threshold_probability(probability):
    probability = float(probability)
    if probability > 0.9:
        probability = probability - 0.1
    elif probability < 0.6:
        probability = probability + 0.1
    return probability


if __name__ == '__main__':
    app.run(debug=True, port=5000)
