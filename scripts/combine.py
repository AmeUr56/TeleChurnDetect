import nbformat

notebooks = ["../notebooks/01_frame_problem.ipynb", "../notebooks/02_data_collection.ipynb", "../notebooks/03_data_exploration.ipynb","../notebooks/04_data_preparation.ipynb","../notebooks/05_shortlist_best_models.ipynb","../notebooks/06_finetune_models.ipynb","../notebooks/07_solution_presentation.ipynb"]

# Create a new notebook object
merged_notebook = nbformat.v4.new_notebook()

# Merge content from each notebook
for notebook_path in notebooks:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
        merged_notebook.cells.extend(nb.cells)

# Save the merged notebook
with open("../notebooks/full_notebook.ipynb", 'w', encoding='utf-8') as f:
    nbformat.write(merged_notebook, f)