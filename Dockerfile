# Use python as base image
FROM python:3.5-slim

# use WORKING DIRECTORY /app
WORKDIR /app

#copy all the content of current directory to /app
ADD . /app

#Installing required packages
RUN pip install --trusted-host pypi.python.org -r requirements.txt

#Open port 5000
EXPOSE 5000

#Set Enviornment variable
ENV NAME OpentoALL

#run python program
CMD ["python","app.py"]

