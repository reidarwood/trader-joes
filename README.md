# trader-joes

To get started with the project first create an environment by running `bash ./create_venv.sh`

After that navigate to the code folder and run: `python main.py --download-data`.
Due to API restrictions this will likely take a long time as only 5 requests per minute are allowed, but will download stocks into `data/stocks` folder all as `.csv` files