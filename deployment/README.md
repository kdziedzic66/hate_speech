# Deployment of Hate Speech

In purpose of deployment simple Flask application with one endpoint has been written. For details of the implementation see `app.py`

For ability to run the deployment independently of the servers system the containarized deployment of solution has been proposed.

To scale up containers to the desired number capable of handling possibly many requests and for managing and versioning the deployment we use Kubernetes. One of the reasons for that is that main cloud services (AWS and GCP for sure, i guess that Azure would also in upcoming future) provide Kuberentes clusters on which the solution can be run

Our system architecture consists of three pods (each one consitst of exactly one container with mentioned Flask application) and one load balancer to handle traffic.


`docker build -f Dockerfile -t your_tag ../`