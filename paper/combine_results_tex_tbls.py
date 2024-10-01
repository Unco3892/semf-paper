import os

def read_latex_table(file_path):
    """Reads the LaTeX table from a file and extracts the rows."""
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    start_index = None
    end_index = None
    for i, line in enumerate(lines):
        if '\\midrule' in line:
            start_index = i + 1
        elif '\\bottomrule' in line:
            end_index = i
            break
    
    table_rows = lines[start_index:end_index]
    return table_rows

def combine_tables_with_titles(table_files, titles, output_file, data_type, remove_old_files=True):
    """Combines the rows of multiple LaTeX tables into one table with titles and horizontal lines spanning all columns.

    Args:
        table_files (list): A list of file paths of the LaTeX tables to be combined.
        titles (list): A list of titles for each table.
        output_file (str): The file path of the output combined table.
        data_type (str): The type of data used in the tables.
        remove_old_files (bool, optional): Whether to remove the original table files after combining. Defaults to True.
    """
    combined_rows = []
    
    for i, file in enumerate(table_files):
        table_rows = read_latex_table(file)
        
        # Remove MPIW column
        table_rows = [remove_mpiw_column(row) for row in table_rows]
        
        if i > 0:
            combined_rows.append(["\\cmidrule(lr){1-9}\n"])
        combined_rows.append([f"\\textbf{{{titles[i]}}} & & & & & & & & \\\\ \\cmidrule(lr){{1-9}}\n"])
        combined_rows.extend(table_rows)
    
    with open(output_file, 'w') as file:
        file.write(f"""
\\begin{{table*}}[t]
\\caption{{Test results for all models with {data_type} data at 95\\% quantiles aggregated over five seeds. For each metric, the mean and standard deviation of the performance across the seeds are separated by $\\pm$. Performance over the baseline is highlighted in bold.}}
\\label{{table-combined-{data_type}-aggregated}}
\\vskip 0.15in
\\begin{{small}}
\\begin{{sc}}
\\centering
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{lcccccccc}}
\\toprule

\\multicolumn{{1}}{{c}}{{}} & \\multicolumn{{5}}{{c}}{{Interval Predictions}} & \\multicolumn{{3}}{{c}}{{Point Predictions}} \\\\
\\cmidrule(lr){{2-6}} \\cmidrule(lr){{7-9}}
 & \\multicolumn{{3}}{{c}}{{Relative}} & \\multicolumn{{2}}{{c}}{{Absolute}} & \\multicolumn{{2}}{{c}}{{Relative}} & Absolute \\\\
\\cmidrule(lr){{2-4}} \\cmidrule(lr){{5-6}} \\cmidrule(lr){{7-8}} \\cmidrule(lr){{9-9}}
Dataset & $\\Delta$CWR & $\\Delta$PICP & $\\Delta$NMPIW & PICP & NMPIW & $\\Delta$RMSE & $\\Delta$MAE & $\\text{{R}}^2$ \\\\
\\midrule
""")
        for row in combined_rows:
            for line in row:
                file.write(line)
        
        file.write("""
\\bottomrule
\\end{tabular}
}
\\end{sc}
\\end{small}
\\end{table*}
""")

    if remove_old_files:
        for file in table_files:
            os.remove(file)
            print(f"Removed file: {file}")

def remove_mpiw_column(row):
    columns = row.split('&')
    if len(columns) == 10:  # Assuming MPIW is the 6th column (index 5)
        del columns[5]
    return ' & '.join(columns)

if __name__ == '__main__':
    
    # Define the data types
    data_types = ["complete", "missing"]
    for data_type in data_types:

        titles = [
            "MultiXGBs",
            "MultiETs",
            "MultiMLPs"
        ]

        # lowercase all model names
        title_lower = [title.lower() for title in titles]

        # Define the directory containing the LaTeX tables
        table_dir = "tex_tbls"

        # Optionally produce raw prediction interval files
        # data_type += "-raw"

        table_files = []
        for title in title_lower:
            table_files.append(os.path.join(table_dir, f"table-{title}-{data_type}-aggregated.tex"))

        # Output file path
        output_file = os.path.join(table_dir, f"table-combined-{data_type}-aggregated.tex")

        # Combine the tables with titles and remove old files if specified
        combine_tables_with_titles(table_files, titles, output_file, data_type, remove_old_files=True)

        print(f"Combined {data_type} table saved to {output_file}")
