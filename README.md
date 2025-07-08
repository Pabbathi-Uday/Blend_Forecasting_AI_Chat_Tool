# Getting Started
Follow these steps to dockerize a Streamlit application:

## Clone this Repository:

### Build the Docker Image:

Run the following command to build a Docker image for your Streamlit application : 
```bash
docker build -t genai-insights .
```
### Run the Docker Container:
Start a Docker container from the image using this command : 
```bash
docker run -p 8501:8501 genai-insights 
```
This command maps port 8501 in the container to port 8501 on your host machine.

### Access the Streamlit Application:
Open a web browser and navigate to 

http://localhost:8501 

to view and interact with the Streamlit app.
