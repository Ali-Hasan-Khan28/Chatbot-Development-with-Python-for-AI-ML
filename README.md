# Chatbot-Development-with-Python-for-AI-ML
# Chatbot Development with Python for AI/ML

This repository contains a chatbot built using **Python, Flask, and OpenAI**, containerized with **Docker** for easy deployment.

## Getting Started

Follow the steps below to set up and run the chatbot on your local machine.

### 1. Clone the Repository

```bash
git clone https://github.com/Ali-Hasan-Khan28/Chatbot-Development-with-Python-for-AI-ML.git
cd Chatbot-Development-with-Python-for-AI-ML
```

### 2. Open the Project in an IDE

Use VS Code, PyCharm, or any other compatible IDE to open the project folder.

### 3. Create a .env File

Inside the project folder, create a new file named .env and add the following content:

```bash
PINECONE_API_KEYS = 'your_apikey'
OPENAI_API_KEY = "your_apikey"
```

Replace 'your_apikey' with your actual API keys.
### 4. Ensure Docker is Installed

Make sure Docker is installed on your system. If not, install it from Docker's official site.
### 5. Build the Docker Image

Run the following command to build the Docker image:
```bash
docker build -t "yourusername"/python-chatbot:latest .
```
Replace "yourusername" with your actual Docker Hub username.
### 6. Run the Docker Container

Once the image is built, start the chatbot container with:
```bash
docker run -d -p 5000:5000 alihasankhan/python-chatbot
```

### 7. Access the Chatbot

Open any browser and visit:
```bash
http://localhost:5000
```
The chatbot should now be running! 🎉

## License

This project is licensed under the MIT License.

```bash
This README file follows standard formatting with proper headings, code blocks, and links. Let me know if you need any modifications! 🚀
```


