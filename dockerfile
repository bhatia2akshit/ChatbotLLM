FROM python:3.9.6

# Set the working directory inside the container
WORKDIR /dockerApp

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies within the virtual environment
RUN pip install torch torchaudio torchvision flask transformers

# Copy the application files into the container
COPY app.py .
COPY templates/frontend.html templates/
COPY static/styling.css static/

# Expose port 5000 for the Flask application
EXPOSE 5000

# Command to run the Flask application within the virtual environment
CMD ["python", "app.py"]
