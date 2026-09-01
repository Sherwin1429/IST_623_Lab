import streamlit as st

# Create pages for each lab
lab1 = st.Page("Lab1.py", title="Lab 1")
lab2 = st.Page("Lab2.py", title="Lab 2", default=True)

# Create navigation
pg = st.navigation([lab1, lab2])

# Run the selected page
pg.run()