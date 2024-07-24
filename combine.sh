for file in app.py bookmark_organizer.py config.py main.py models.py routes.py; do
    echo -e "\n--- Start of $file ---\n"; 
    cat "$file"; 
    echo -e "\n--- End of $file ---\n"; 
done > combined_python_files.txt