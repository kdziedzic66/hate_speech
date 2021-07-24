FROM python:3.6

WORKDIR /code

COPY ./ /code

RUN pip install -r requirements.txt
RUN pip install -e .

EXPOSE 5000

ENTRYPOINT ["python3"]
CMD ["deployment/app.py"]