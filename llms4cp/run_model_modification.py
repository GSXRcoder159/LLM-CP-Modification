import argparse
import os
from datetime import datetime

from llms4cp.data_reader import read_model_modification_examples
from llms4cp.model_modification_pipeline import run_model_modification_pipeline
from llms4cp.in_context_config import MODEL

def main():
    """
    Main entry point for running the model modification pipeline.
    """
    parser = argparse.ArgumentParser(description='Run the constraint model modification pipeline')
    parser.add_argument('--method', type=str, default='CPMPY', choices=['CPMPY', 'PSEUDO', 'NER'],
                        help='Method to use for model modification')
    parser.add_argument('--dataset', type=str, default='APLAI', choices=['APLAI', 'NL4OPT', 'LGPs'],
                        help='Dataset to use for model modification')
    parser.add_argument('--data_path', type=str, default='data/modification/APLAI_course',
                        help='Path to the directory containing model modification examples')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit the number of examples to process')
                        
    args = parser.parse_args()
    
    # Set environment variable for limit if specified
    if args.limit:
        os.environ['LIMIT'] = str(args.limit)
    
    # Load model modification examples
    examples = read_model_modification_examples(args.data_path)
    
    print(f"Running model modification pipeline with method: {args.method}")
    print(f"Using model: {MODEL}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Number of examples: {len(examples)}")
    
    # Run the pipeline
    run_model_modification_pipeline(examples, args.dataset, args.method)
    
if __name__ == '__main__':
    main()