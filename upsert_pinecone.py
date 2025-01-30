from dotenv import load_dotenv
import os
from src.vectordb_utils import import_csv_to_vector, index

# Load environment variables at the start
load_dotenv()

def main():
    try:
        # Import different types of data into vector database
        import_csv_to_vector(os.path.join("fixture", "info.csv"))
        # Print index statistics
        print("Vector Database Statistics:")
        print(index.describe_index_stats())
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()