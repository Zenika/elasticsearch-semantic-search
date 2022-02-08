import tensorflow as tf
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from functional import seq


def define_sentences_index(client: Elasticsearch, delete_index: bool, index_name: str, number_of_hash_tables: int,
                           embedding_vector_dim: int):
    if delete_index or not _index_exists(client, index_name):
        client.indices.delete(index_name, ignore=[404], request_timeout=60)
        index_definition = {
            "settings": {
                "number_of_shards": 3,
                "number_of_replicas": 0
            },
            "mappings": {
                "properties": {
                    "sentence": {
                        "type": "text", "index": False
                    },
                    "sentence_vector": {
                        "type": "dense_vector",
                        "dims": embedding_vector_dim
                    }
                }
            }
        }
        for i in range(number_of_hash_tables):
            index_definition["mappings"]["properties"][_lsh_field_name(i)] = {
                "type": "integer"
            }
        print("create index", index_name, "definition", index_definition)
        client.indices.create(index=index_name, body=index_definition)


def define_hash_tables_index(client: Elasticsearch, index_name: str, delete_index: bool, number_of_hash_tables: int,
                             number_of_plans: int, embedding_vector_dim: int):
    if delete_index or not _index_exists(client, index_name):
        partitions = tf.random.uniform((number_of_hash_tables, number_of_plans, embedding_vector_dim), -1, 1, seed=0)
        _store_hash_tables(client, partitions, index_name)


def _lsh_field_name(i):
    return "hash_{:03d}".format(i)


def _index_exists(client: Elasticsearch, index_name: str):
    return client.indices.exists(index_name)


def _store_hash_tables(client: Elasticsearch, hash_tables: tf.Tensor, table_index_name: str = "hash_tables"):
    client.indices.delete(table_index_name, ignore=[404])
    number_of_hash_tables, number_of_plans, d = hash_tables.shape
    normal_vectors = {}
    plan_formater = 'p_{:02d}'
    for plan in range(0, number_of_plans):
        normal_vectors[plan_formater.format(plan)] = {
            "type": "dense_vector",
            "dims": d
        }
    index_definition = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        },
        "mappings": {
            "properties": {
                "hash_table_number": {
                    "type": "integer"
                },
                "normal_vectors": {
                    "properties": normal_vectors
                }
            }
        }
    }
    client.indices.create(index=table_index_name, body=index_definition)
    for hash_table_number in range(number_of_hash_tables):
        normal_vectors_doc = {}
        for plan in range(0, number_of_plans):
            normal_vectors_doc[plan_formater.format(plan)] = hash_tables[hash_table_number][plan].numpy()
        document = {
            "hash_table_number": hash_table_number,
            "normal_vectors": normal_vectors_doc
        }
        client.index(index=table_index_name, body=document)


def read_hash_tables(client: Elasticsearch, tables_index_name: str = "hash_tables") -> tf.Tensor:
    client.indices.refresh(index=tables_index_name)
    res = client.search(index=tables_index_name,
                        body={
                            "query": {
                                "match_all": {}
                            },
                        },
                        params={
                            "size": 1000
                        })
    hash_tables = seq(res['hits']['hits']) \
        .map(lambda h: h['_source']) \
        .order_by(lambda src: src['hash_table_number']) \
        .map(lambda src: src['normal_vectors']) \
        .map(lambda nvs: seq(nvs.items())
             .sorted()
             .map(lambda kv: kv[1])
             .list()) \
        .list()
    return tf.convert_to_tensor(hash_tables)


def _gen_index_doc(sentences: tf.Tensor, embedings: tf.Tensor, lshs: tf.Tensor,
                   index_name: str):
    all_docs = zip(sentences.numpy(), embedings.numpy(), tf.transpose(lshs).numpy())
    for i, doc_elements in enumerate(all_docs):
        sentence, sentence_vector, hashes = doc_elements
        doc = dict(
            {'_op_type': 'index', "_index": index_name,
             "sentence": sentence.decode(), "sentence_vector": sentence_vector},
            **dict([(_lsh_field_name(i), val) for i, val in enumerate(hashes)]))
        yield doc


def index_sentences(client: Elasticsearch, sentences: tf.Tensor, embedings: tf.Tensor, lshs: tf.Tensor,
                    index_name: str  # = "lsh_sentences_2"
                    ):
    res = bulk(client, _gen_index_doc(sentences, embedings, lshs, index_name), timeout="60s", max_retries=2)
    client.indices.refresh(index=index_name)
    print("index operation done", sentences.shape[0], "sentences", "index", index_name, "res", res)


def query_with_lsh(client: Elasticsearch, lshs: tf.Tensor, encoding: tf.Tensor, index_name: str = "lsh_sentences_2",
                   full_scan=False, number_of_partitions: int = 0, result_count: int = 20, min_score: int = 0,
                   profile: bool = False):
    if full_scan:
        query = {"match_all": {}}
    else:
        query = _get_lsh_query(lshs, number_of_partitions)
    script_query = {
        "script_score": {
            "query": query,
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'sentence_vector') + 1.0",
                "params": {"query_vector": encoding.numpy()[0]}
            }
            , "min_score": min_score
        }
    }
    body = {
        "profile": profile,
        "size": result_count,
        "query": script_query,
        "_source": "sentence",
    }
    timeout = 120 if full_scan else 60
    response = client.search(
        index=index_name,
        body=body,
        request_timeout=timeout
    )
    print(response["took"], "ms", response["hits"]["total"], "hits")
    if profile:
        print("profile", response["profile"])
    return list(map(
        lambda hit:
        {"sentence": str(hit["_source"]["sentence"]), "score": hit["_score"], "id": hit["_id"]},
        response['hits']['hits']))


def _get_lsh_query(lshs, number_of_partitions: int = 0):
    terms = [{"term": {_lsh_field_name(i): val.numpy()[0]}} for i, val in enumerate(lshs)]
    if number_of_partitions > 0:
        query = {"bool": {"should": terms[0:number_of_partitions]}}
    else:
        query = {"bool": {"should": terms}}
    query = {"bool": {"filter": query}}
    return query


def count_with_lsh(client: Elasticsearch, lshs: tf.Tensor, number_of_partitions: int = 0,
                   index_name: str = "lsh_sentences_2"):
    body = {"query": _get_lsh_query(lshs, number_of_partitions)}
    print(str(body).replace("'", "\""))
    return client.count(body=body, index=index_name)
