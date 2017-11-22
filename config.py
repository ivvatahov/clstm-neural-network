class Config:

    # input
    NUM_CLASSES = 2
    EMBEDDING_DIM = 100

    # model
    NUM_UNITS= 32
    FILTER_NUM = 64
    FILTER_SIZE = 3
    MAX_SEQUENCE_LENGTH = 20

    # BUCKETS = [(16, 19)]

    # datasets
    LOGS_PATH = "/app/tmp/logs/10/"
    DATA_ROOT = "/app/data/datasets/amazon-fine-food-reviews/"

    TRAIN_FILENAME = "train_Reviews"
    TEST_FILENAME = "test_Reviews"
    VALID_FILENAME = "valid_Reviews"

    DEFAULT_VOCABULARY_SIZE = 20000

    # training
    LOG_INTERVAL = 100
    NUM_EPOCHS = 20
    BATCH_SIZE = 64
    LEARNING_RATE = 0.0001
    DROPOUT = 0.3
