class Config:

    # input
    num_classes = 2
    embedding_dim = 100

    # model
    num_units = 32
    filter_num = 64
    filter_size = 3

    # datasets and checkpoints
    LOGS_PATH = "/app/tmp/logs/1/"
    DATA_ROOT = "/app/data/datasets/amazon-fine-food-reviews/"

    TRAIN_FILENAME = "train_Reviews"
    TEST_FILENAME = "test_Reviews"
    VALID_FILENAME = "valid_Reviews"

    MODEL_CHECKPOINTS = "/app/checkpoints/1/"

    # training
    log_interval = 100
    num_epochs = 10
    batch_size = 32
    learning_rate = 0.0001
    dropout = 0.3