In this repository, you will find the code related to the series of articles in Zenika's blog:
https://blog.zenika.com/2022/01/13/de-lalgorithme-a-la-solution-un-exemple-dexploitation-dun-algorithme-dintelligence-artificielle-1/

## Python setup

In order to have a working environment, you will have to perform the following actions:
- initialize the virtual environment by typing the command `virtualenv venv` in this directory
- install dependencies by typing the command `pip install -r requirements.txt`

## Loading dataset
We provide `dataset/load_dataset.py` script. Run this script once.

## Running Elasticsearch server

`docker compose up -d` will launch a 3 node cluster on your machine. 
In order to stop the cluster, type `docker compose stop` 

Alternatively, you can type `docker compose -f docker-compose-single-node.yml` if you prefer a single node
setup
In order to stop the cluster, type `docker compose -f docker-compose-single-node.yml stop` 

## Indexing data

In a shell, type `python CorpusIndexer.py` in project directory. You will ingest some data from
the `para_crawl/enfr` corpus which contains sentences or small paragraphs in english and
french. 

The first execution is quite long, as the dataset, which contains more than 30 millions documents,
will be downloaded to your machine. After that, indexation is starting

When you feel you have enough documents for your experiments, you can stop the process 
by hitting `Ctrl-C` in the shell console. 

## Search data

In a shell, type `python CorpusQuery.py` in project directory. This is a very basic command line interface application
to perform queries. Simply type your query and the application will return results that are semantically close
to your query and ordered by semantic relevance

To stop the application, hit `Ctrl-C` in the shell console. 
