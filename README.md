# Deploying a Prediction App on Google Cloud Platform (GCP)

The App Engine is a cloud-based platform, is quite comprehensive, and combines infrastructure as a service (IaaS), platform as a service (PaaS), and software as a service (SaaS). The runtime and languages are up-to-date, with great documentation. Features in the preview stage (beta) are made available to a large number of users, which keeps us informed of possible future developments.
To deploy this model on the App Engine using a terminal, there are four major things needed, which are:

•	The serialized model and model artifacts: This is the saved trained model and other standard objects used during data transformation. All will be stored in Google Storage (bucket) upon deployment so that they can be accessible by the main script for test data preparation and making predictions.

•	Main script.py: This is the script in which the prediction function is written, and where all necessary libraries listed in the requirements file needed for end-to-end data preparation and prediction are imported. I would add comments for each line of code so that it’s easier to read.

•	Requirement.txt: A simple text file that contains model dependencies with the exact version used during the training of the model. In order to avoid running into trouble, it’s better to check available versions of all the libraries and packages you’ll be using on the cloud before developing the model.

scikit-learn==0.22
numpy==1.18.0
pandas==0.25.3
flask==1.1.1

•	The app.yaml file: This is the file you can use to configure your App Engine app’s settings in the app.yaml file. This file specifies how URL paths correspond to request handlers and static files. The app.yaml file also contains information about your app's code, such as the runtime and the latest version identifier. Any configuration you omit on this file will be set to the default state. For this simple app, I only need to set the run time to Python37 so that the App Engine can know the Docker image that will be running the app.
runtime: python37
We need to have the project on a software development version control platform such as Bitbucket, GitHub, etc. I have chosen GitHub.

### Steps to deploying the model on Google’s App Engine

•	Create a project on the Google Cloud Platform

•	Select the project and create an app using App Engine. Set up the application by setting the permanent region where you want Google to manage your app. After this step, select the programming language used in writing the app (Python).

•	I’m using cloud shell. once the shell is activated, ensure that the Cloud Platform project is set to the intended project ID.

•	Clone your GitHub project repo on the engine by running (git clone <link to clone your repository> )

•	Change to the directory of the project containing the file to be uploaded on App Engine by running (cd ‘ cloned project folder’). You can call directories by running ls.

•	Initialize gcloud in the project directory by running gcloud init. This will trigger some questions on the configuration of the Google Cloud SDK, which are pretty straightforward and can be easily answered.

•	The last step is to deploy the app by running the command gcloud app deploy. It will take some time to upload files, install app dependencies, and deploy the app.

•	Once the uploading is done, you can run gcloud app browse to start the app in the browser—or copy the app URL manually if the browser isn’t detected.

•	You need to add the API endpoint to the URL if you’re using a custom prediction routine. For this project, Flask was the choice of web framework and the endpoint was declared as /prediction_endpoint.

### Test app with Postman

Since we didn't build any web interface for the project, we can use the Google app client to send an HTTP request to test the app, but we’ll use Postman to test it because we’re predicting in batches based on how we’re reading the dataset on the backend. Below is the response from the app after sending an HTTP request to get predictions for the uploaded test data.
![image](https://github.com/Zubiaaa/Deploying-Machine-Learning-Models-on-Google-Cloud-Platform/assets/67237943/7ce9aaf2-99bd-4a55-afaa-063f4d4f7298)



