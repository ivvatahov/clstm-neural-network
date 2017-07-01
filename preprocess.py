from data.data_loader import DataLoader
from data.preprocessor import Preprocessor
                              
preprocessor = Preprocessor('/app/data/datasets/amazon-fine-food-reviews/', 
	                         'Reviews', DataLoader.DEFAULT_VOCABULARY_SIZE)
print("Read...")
preprocessor.read_data()
print("Preprocessing...")
preprocessor.preprocess(data_column='Text', label_column='Score')
print('Saving...')
preprocessor.save_data()
print("Done!")