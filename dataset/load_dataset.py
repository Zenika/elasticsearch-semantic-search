# run this script once to load the dataset on your machine.
# the data will be stored under $HOME/tensorflow_datasets/para_crawl/enfr
import resource
import tensorflow_datasets as tfds

# The load operation may fail with a 'Too many open files' error
# so we change limits
low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2000, high))
try:
    tfds.load("para_crawl/enfr")
finally:
    # restore previous limits
    resource.setrlimit(resource.RLIMIT_NOFILE, (low, high))
