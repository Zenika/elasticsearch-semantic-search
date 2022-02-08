from time import time

from elasticsearch import Elasticsearch

from commons.Constants import SENTENCES_INDEX, HASH_TABLES_INDEX
from es.esFunctions import query_with_lsh, count_with_lsh, read_hash_tables
from lsh.hashComputation import multi_tf_lsh
from sentence_encoder.SentenceEncoder import encode_sentence

client = Elasticsearch()
hash_tables = read_hash_tables(client, HASH_TABLES_INDEX)
number_of_partitions = 20
full_scan = False
min_score = 0
while True:
    query: str = input("search: \n > ")
    if query.strip() == "":
        exit()
    t = time()
    encodings = encode_sentence(query)
    encoding_time = time() - t
    t = time()
    lshs = multi_tf_lsh(hash_tables, encodings)
    lsh_time = time() - t
    t = time()
    t_count = time()
    if not full_scan:
        print("count",
              count_with_lsh(client, lshs, number_of_partitions=number_of_partitions, index_name=SENTENCES_INDEX)["count"])
    t_count = time() - t_count
    results = query_with_lsh(client, lshs, encodings, index_name=SENTENCES_INDEX, full_scan=full_scan,
                             number_of_partitions=number_of_partitions, min_score=min_score)
    query_time = time() - t
    print("encoding", encoding_time, "lsh", lsh_time, "query", query_time, "count", t_count)
    for res in results:
        print(res["sentence"], "(", res["score"], ")", " id:", res["id"])
