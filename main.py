# import argparse

# from llms4cp.aplai_pipeline import run_aplai_pipeline
# from llms4cp.nl4opt_pipeline import run_nl4opt
# from llms4cp.puzzles_pipeline import run_lgp_pipeline


# def pipeline():
#     parser = argparse.ArgumentParser(description="Run CP modelling methods on datasets.")

#     # Add arguments for the dataset and method
#     parser.add_argument('--dataset', choices=['nl4opt', 'LGPs', 'APLAI'], required=True, help='Choose a dataset')
#     parser.add_argument('--method', choices=['DIRECT', 'CPMPY', 'PSEUDO', 'NER'], required=True, help='Choose a method')

#     # Parse the arguments
#     args = parser.parse_args()

#     # Run the chosen dataset with the chosen method
#     if args.dataset == 'nl4opt':
#         run_nl4opt(args.method)
#     elif args.dataset == 'LGPs':
#         run_lgp_pipeline(args.method)
#     elif args.dataset == 'APLAI':
#         run_aplai_pipeline(args.method)
#     else:
#         raise ValueError(f"Dataset {args.dataset} not supported")


# if __name__ == '__main__':
#     pipeline()


import argparse
import os
from datetime import datetime

from llms4cp.data_reader import read_model_modification_examples, read_model_modification_example
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
    parser.add_argument('--output_path', type=str, default='data/modification/APLAI_course/output',
                        help='Path to the directory to save output files')
    parser.add_argument('--input', type=str, default='data/modification/APLAI_course/input.jsonl',
                        help='Path to the input JSONL file containing model modification examples')
    parser.add_argument('--output', type=str, default='data/modification/APLAI_course/output.jsonl',
                        help='Path to the output JSONL file to save results')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit the number of examples to process')
                        
    args = parser.parse_args()
    
    # Set environment variable for limit if specified
    if args.limit:
        os.environ['LIMIT'] = str(args.limit)
    
    if args.input:
        examples = read_model_modification_example(args.input)
    elif args.data_path:
        examples = read_model_modification_examples(args.data_path)
    else:
        print("Warning: No input or data path provided. Exiting.")
        return
    
    all_examples = read_model_modification_example("../data/modification/APLAI_course/all_context_examples.jsonl")
    
    print(f"Running model modification pipeline with method: {args.method}")
    print(f"Using model: {MODEL}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Number of examples: {len(examples)}")
    
    # Run the pipeline
    run_model_modification_pipeline(examples, all_examples, args.dataset, args.method)
    
if __name__ == '__main__':
    main()