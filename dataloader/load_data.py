import os
import pandas as pd

from config import DATA_DIR

# Creating an abstract base class for data loading.
# This allows us to easily swap out different data sources in the future (e.g., databases, APIs) without changing the rest of our codebase. 
# For now, we have a concrete implementation that loads from a CSV file, but we could easily add new loaders that read from other sources by subclassing BaseDataLoader and implementing the load_data method.
class BaseDataLoader:
    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = data_dir

    def load_data(self):
        raise NotImplementedError("Subclasses must implement load_data()")
    

class FraudDataLoader(BaseDataLoader):
    def load_data(self, filename="applications_v1.csv"):
        data_path = os.path.join(self.data_dir, filename)
        return pd.read_csv(data_path)
    

