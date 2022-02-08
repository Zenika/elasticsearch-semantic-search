import math
import time

import tensorflow as tf
import tensorflow_datasets as tfds
from elasticsearch import Elasticsearch

from commons.Constants import HASH_TABLES_INDEX, SENTENCES_INDEX
from es.esFunctions import define_sentences_index, index_sentences, define_hash_tables_index, read_hash_tables
from lsh.hashComputation import multi_tf_lsh
from sentence_encoder.SentenceEncoder import encode_sentence

EMBEDDING_VECTOR_DIM = 512

number_of_sentences_pairs_to_index = 2_000_000
delete_index = True

# define dimensions
number_of_sentences_to_index = number_of_sentences_pairs_to_index * 2
avg_item_per_partition = 500
# heuristic that estimate the number of (hyper)plans in order to have approximately the number of items (vectors)
# per partition. One plan divides space in 2 regions
number_of_plans = math.ceil(math.log2(number_of_sentences_to_index / avg_item_per_partition))
number_of_hash_tables = 20

client = Elasticsearch()
# define ES indices
define_sentences_index(client, delete_index, SENTENCES_INDEX, number_of_hash_tables, EMBEDDING_VECTOR_DIM)
define_hash_tables_index(client, HASH_TABLES_INDEX, delete_index, number_of_hash_tables, number_of_plans,
                            EMBEDDING_VECTOR_DIM)

# retrieve hash tables
partitions = read_hash_tables(client, HASH_TABLES_INDEX)

# load sentences to index
batch_size = 5000
number_of_batches = int(number_of_sentences_pairs_to_index / batch_size)
ds = tfds.load("para_crawl/enfr_plain_text", batch_size=batch_size)
ds = ds['train']
ds = ds.take(number_of_batches)

indexation_count = 0
number_of_indexations_to_ignore = 0
# for each batch
for batch in ds:
    for lang in ['fr', 'en']:
        indexation_count += 1
        if indexation_count > number_of_indexations_to_ignore:
            sentences = batch[lang]
            sentences = map(lambda s: s.numpy().decode('utf-8'), sentences)
            sentences = tf.convert_to_tensor(list(sentences))
            encodings = encode_sentence(sentences)
            lshs = multi_tf_lsh(partitions, encodings)
            ko = True
            attempt = 1
            sleep_s = 1
            while ko:
                try:
                    index_sentences(client, sentences, encodings, lshs, SENTENCES_INDEX)
                    ko = False
                except Exception as error:
                    sleep_s = sleep_s * 2
                    print("sleep", sleep_s, "sec.")
                    time.sleep(sleep_s)
                    attempt += 1
                    if attempt > 7:
                        raise error
        else:
            print("ignored index operation", batch_size, "sentences")
