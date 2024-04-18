from train import *
from test import *
import argparse
import csv


if __name__ == "__main__":
    main_parser = argparse.ArgumentParser()
    main_parser.add_argument('-input_csv', help='Path to CSV describing experiments to run')
    main_parser.add_argument('-output_csv', help='Path to CSV for tracking experiment results')
    args = main_parser.parse_args()

    INPUT_CSV_FILENAME = args.input_csv
    OUTPUT_CSV_FILENAME = args.output_csv
    results = []

    # Use experiment CSV file to train/test on specific audio inputs.
    with open(INPUT_CSV_FILENAME, newline='') as file:
        reader = csv.DictReader(file)
        training_parser = training_parser()
        testing_parser = testing_parser()
        
        for row in reader:

            # Append experiment.csv-specified training arguments
            training_arg_list = []
            for k, v in row.items():
                if k == "text":
                    pass  # Ignore the text argument for now, it will be used for testing
                else:
                    training_arg_list.append("-" + k)  # Dash for argument name

                    if k == "content" or k == "style":  # Need to append input path to the file name
                        training_arg_list.append("input/" + v)
                    else:
                        training_arg_list.append(v)

            # Specify output filename based on content file name
            content_id = row["content"].split(".")[0]
            training_arg_list.append("-output")
            training_arg_list.append("output/" + content_id)
                
            training_args = training_parser.parse_args(training_arg_list)
            train_model(training_args)

            # Create testing arguments here, as they do not need to be specified in the csv
            testing_arg_list = [
                "-original", "input/" + row["content"],
                "-original_start", row["content_start"],
                "-original_end", row["content_end"],
                "-transformed", "output/" + content_id + ".wav",
                "-transformed_start", row["content_start"],
                "-transformed_end", row["content_end"],
                "-text", row["text"]
            ]

            testing_args = testing_parser.parse_args(testing_arg_list)
            original_wer, original_conf, transformed_wer, transformed_conf = test_model_output(testing_args)

            results.append([content_id, original_wer, original_conf, transformed_wer, transformed_conf])

    # Store results in output CSV file
    with open(OUTPUT_CSV_FILENAME, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["accent", "original WER", "original confidence", "transformed WER", "transformed confidence"])
        writer.writerows(results)