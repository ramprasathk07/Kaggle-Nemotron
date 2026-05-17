import pandas as pd

def main():
    input_file = 'data/generated_cot/train_v15_fixed.csv'
    output_file = 'data/generated_cot/new_COT.csv'
    
    print(f"Reading from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Sort the dataframe by 'puzzle_type'
    df_sorted = df.sort_values(by='puzzle_type')
    
    # Filter out 250 rows.
    # Filter out 250 rows per puzzle type
    df_filtered = df_sorted.groupby('puzzle_type').head(250)
    
    # Keep only the 'prompt' and 'answer' columns
    final_df = df_filtered[['prompt', 'answer']]
    
    # Save the result
    final_df.to_csv(output_file, index=False)
    print(f"Successfully saved {len(final_df)} rows to {output_file}")

if __name__ == "__main__":
    main()
