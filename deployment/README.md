# Deployment of Hate Speech

In purpose of deployment simple Flask application with one endpoint has been written. For details of the implementation see `app.py`. This system works on CPU only. 

For ability to run the deployment independently of the servers system the containarized deployment of solution has been proposed.

To scale up containers to the desired number capable of handling possibly many requests and for managing and versioning the deployment we use Kubernetes. One of the reasons for that is that main cloud services (AWS and GCP for sure, i guess that Azure would also in upcoming future) provide Kuberentes clusters on which the solution can be run which makes possibility of scalable public deployment very easy.

Our system architecture consists of three pods (each one consitst of exactly one container with mentioned Flask application) and one load balancer to handle traffic.

Performance tests of the application are printed out inside `test_deployment.ipynb` notebook. It has been done using Minikube cluster.


# Running instructions on Minikube:

It is preassumed that you have docker, minikube and kubectl installed.

- To rebuild (update) the docker image run:

    `docker build -f Dockerfile -t your_tag ../`
    
    `docker tag your_tag kdziedzic66/hatespeech`

- Login to dockerhub:

    `docker login -u "$USER_NAME` -p "$PASSWORD"
  
- Push the image:
    `docker push kdziedzic66/hatespeech`
  
- Create deployment using `deployment.yaml` file:
    `kubectl create -f deployment.yaml`
  To check the status of the deployment run: `kubectl rollout status deployment service`
  After you see: "deployment "service" successfully rolled out" the service would be ready to use.
  To update deployment run `kubectl apply -f deployment.yaml`
  
- Create load balancer using `load_balancer.yaml`:
    `kubectl create -f load_balancer.yaml`
  
- Get the service url by running `minikube service list`
  
