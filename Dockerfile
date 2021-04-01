FROM carlduke/eidos-base:latest

# Install deps
COPY requirements.txt /requirements.txt
RUN pip3 install -r /requirements.txt

# Copy main.py
COPY main.py /main.py

CMD ["python3", "/main.py"]