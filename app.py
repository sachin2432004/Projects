import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
import base64
from datetime import datetime

# Set the page configuration
st.set_page_config(
    page_title='DataPlotter: Instant Plot Generator for Multiple Datasets', 
    layout='wide', 
    page_icon='📊'
)

# Inject custom CSS for background color and font type
def add_bg_from_url():
    bg_url = "https://images.unsplash.com/photo-1557683316-973673baf926?q=80&w=2029&auto=format&fit=crop"
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url('{bg_url}');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
    """, unsafe_allow_html=True)

# Add a dark overlay for readability
def add_overlay():
    st.markdown("""
        <style>
        .stApp:before {
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.7);
            z-index: -1;
        }
        </style>
    """, unsafe_allow_html=True)

# Inject styling
def inject_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600&display=swap');
        html, body, [class*="css"] {
            font-family: 'Montserrat', sans-serif;
            color: #FFFFFF;
        }
        h1,h2,h3 {
            color: #00BFFF;
            font-weight: 600;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
        }
        .highlight-container {
            background-color: #001f3f;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 15px;
            border-left: 5px solid #ffcba4;
        }
        .plot-container {
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .uploadedFile {
            background-color: rgba(0, 191, 255, 0.1) !important;
            border: 1px solid rgba(0, 191, 255, 0.3) !important;
            border-radius: 5px !important;
            padding: 10px !important;
        }
        .stRadio > label {
            background-color: rgba(0, 0, 0, 0.4);
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        .download-btn {
            background-color: #00BFFF;
            color: white;
            padding: 10px 15px;
            border-radius: 5px;
            text-decoration: none;
            display: inline-block;
            margin-top: 10px;
            font-weight: bold;
            border: none;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        .download-btn:hover {
            background-color: #0099CC;
        }
        </style>
    """, unsafe_allow_html=True)

# Highlighted text box
def highlighted_text(text, element_type="div"):
    st.markdown(f"""
        <{element_type} class="highlight-container">
        {text}
        </{element_type}>
    """.strip(), unsafe_allow_html=True)

# Container for plots
def plot_container(plot_function):
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    plot_function()
    st.markdown('</div>', unsafe_allow_html=True)

# Function to create a download link for the plot
def get_image_download_link(fig, filename, format="png", dpi=300):
    buf = io.BytesIO()
    fig.savefig(buf, format=format, dpi=dpi, bbox_inches='tight', transparent=True)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:image/{format};base64,{b64}" download="{filename}.{format}" class="download-btn">Download {filename}</a>'
    return href

# Apply style functions
add_bg_from_url()
add_overlay()
inject_custom_css()

st.title('⚡ Data Plotter: Instant Plot Generator')

highlighted_text("<h3>Upload your CSV file(s)</h3><p>Select up to two CSV files to analyze and generate visualizations</p>")

# Create two upload columns for up to two files
col1, col2 = st.columns(2)

with col1:
    uploaded_file1 = st.file_uploader("Choose first CSV file", type="csv", key="file1")

with col2:
    uploaded_file2 = st.file_uploader("Choose second CSV file (optional)", type="csv", key="file2")

# Process uploaded files
if uploaded_file1 is not None:
    try:
        # Read the first file
        df1 = pd.read_csv(uploaded_file1, keep_default_na=True, na_values=['NA', 'N/A', 'None', 'none', ''])
        
        # Replace string 'None' values with actual NaN
        df1.replace(['None', 'none', 'NaN', 'nan', 'NA', 'N/A', ''], np.nan, inplace=True)
        
        # Show file info
        st.write(f"File 1: {uploaded_file1.name} - {df1.shape[0]} rows, {df1.shape[1]} columns")
        
        # Check if second file exists
        if uploaded_file2 is not None:
            try:
                df2 = pd.read_csv(uploaded_file2, keep_default_na=True, na_values=['NA', 'N/A', 'None', 'none', ''])
                df2.replace(['None', 'none', 'NaN', 'nan', 'NA', 'N/A', ''], np.nan, inplace=True)
                st.write(f"File 2: {uploaded_file2.name} - {df2.shape[0]} rows, {df2.shape[1]} columns")
                
                # Choose how to combine datasets
                combine_method = st.radio(
                    "How would you like to combine the datasets?",
                    ["Merge on common column", "Concatenate (stack rows)"]
                )
                
                if combine_method == "Merge on common column":
                    # Get common columns for join
                    common_cols = list(set(df1.columns).intersection(set(df2.columns)))
                    if not common_cols:
                        st.warning("No common columns found for joining datasets. Using concatenation instead.")
                        df = pd.concat([df1, df2], ignore_index=True)
                        st.success(f"Successfully concatenated datasets! New shape: {df.shape[0]} rows, {df.shape[1]} columns")
                    else:
                        join_col = st.selectbox("Select column to join on:", common_cols)
                        join_how = st.selectbox("Select join type:", 
                                              ["outer (keep all rows)", 
                                               "inner (keep only matching rows)", 
                                               "left (keep all rows from first file)",
                                               "right (keep all rows from second file)"])
                        
                        # Convert join_how to actual parameter
                        join_map = {
                            "outer (keep all rows)": "outer", 
                            "inner (keep only matching rows)": "inner",
                            "left (keep all rows from first file)": "left",
                            "right (keep all rows from second file)": "right"
                        }
                        
                        # Create a progress message
                        with st.spinner("Merging datasets..."):
                            # Merge datasets
                            df = pd.merge(df1, df2, on=join_col, how=join_map[join_how], suffixes=('', '_y'))
                            
                            # Get list of columns ending with '_y'
                            y_columns = [col for col in df.columns if col.endswith('_y')]
                            
                            # For each '_y' column, fill NaN values in the original column with values from '_y'
                            for col in y_columns:
                                base_col = col[:-2]  # Remove '_y' suffix
                                # Fill NaN values in base column with values from y column
                                df[base_col] = df[base_col].fillna(df[col])
                                # Drop the redundant '_y' column
                                df = df.drop(col, axis=1)
                            
                            st.success(f"Successfully joined datasets on '{join_col}'! New shape: {df.shape[0]} rows, {df.shape[1]} columns")
                else:  # Concatenate
                    with st.spinner("Concatenating datasets..."):
                        # Get all columns from both dataframes to ensure complete schema
                        all_columns = list(set(df1.columns) | set(df2.columns))
                        
                        # Make sure both dataframes have the same columns before concatenation
                        for col in all_columns:
                            if col not in df1:
                                # Find the most common non-null value in df2[col] to use as default
                                if df2[col].notna().any():
                                    default_val = df2[col].value_counts().index[0] if not df2[col].value_counts().empty else 0
                                else:
                                    default_val = 0 if df2[col].dtype.kind in 'iuf' else ""  # numeric or string default
                                df1[col] = default_val
                            
                            if col not in df2:
                                # Find the most common non-null value in df1[col] to use as default
                                if df1[col].notna().any():
                                    default_val = df1[col].value_counts().index[0] if not df1[col].value_counts().empty else 0
                                else:
                                    default_val = 0 if df1[col].dtype.kind in 'iuf' else ""
                                df2[col] = default_val
                        
                        # Now concatenate with complete columns in both dataframes
                        df = pd.concat([df1, df2], ignore_index=True)
                        
                        # Auto-convert None/NaN string values to proper NaN for easier handling
                        df.replace(['None', 'none', 'NaN', 'nan', 'NA', 'N/A', ''], np.nan, inplace=True)
                        
                        st.success(f"Successfully concatenated datasets! New shape: {df.shape[0]} rows, {df.shape[1]} columns")
            except Exception as e:
                st.error(f"Error reading the second CSV file: {e}")
                df = df1  # Fallback to first dataset
        else:
            # Use only first dataset
            df = df1
        
        # Convert any string representation of None to actual NaN values
        df.replace(['None', 'none', 'NaN', 'nan', 'NA', 'N/A', ''], np.nan, inplace=True)
        
        # Handle missing values
        if df.isnull().any().any():
            missing_count = df.isnull().sum().sum()
            st.warning(f"Dataset contains {missing_count} missing values")
            
            # Auto-suggest the best handling method based on data characteristics
            suggested_method = "Fill all missing values with most frequent values"
            if missing_count / (df.shape[0] * df.shape[1]) > 0.5:  # If more than 50% is missing
                suggested_method = "Fill missing numerical values with median"
            
            missing_handling = st.selectbox(
                "How would you like to handle missing values?",
                ["Fill all missing values with most frequent values",  # Move this to first position as default
                 "Fill missing numerical values with mean",
                 "Fill missing numerical values with median",
                 "Fill missing categorical values with mode",
                 "Fill with zeros",
                 "Drop rows with any missing values", 
                 "Keep as is"]
            )
            
            if missing_handling == "Drop rows with any missing values":
                original_rows = df.shape[0]
                df = df.dropna()
                st.info(f"Dropped {original_rows - df.shape[0]} rows with missing values")
            
            elif missing_handling == "Fill missing numerical values with mean":
                numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                for col in numeric_cols:
                    df[col] = df[col].fillna(df[col].mean())
                st.info(f"Filled missing numerical values with mean")
            
            elif missing_handling == "Fill missing numerical values with median":
                numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                for col in numeric_cols:
                    df[col] = df[col].fillna(df[col].median())
                st.info(f"Filled missing numerical values with median")
            
            elif missing_handling == "Fill missing categorical values with mode":
                categorical_cols = df.select_dtypes(include=['object']).columns
                for col in categorical_cols:
                    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
                st.info(f"Filled missing categorical values with mode")
                
            elif missing_handling == "Fill all missing values with most frequent values":
                # For each column, fill with the most frequent value
                missing_before = df.isnull().sum().sum()
                
                for col in df.columns:
                    if df[col].isnull().any():  # Only process columns with missing values
                        if df[col].dtype == 'object' or df[col].dtype.name == 'category':  # Categorical
                            # Try to use mode, fallback to "Unknown"
                            if not df[col].mode().empty:
                                fill_val = df[col].mode()[0]
                            else:
                                fill_val = "Unknown"
                            df[col] = df[col].fillna(fill_val)
                        else:  # Numerical
                            # For numerical columns with very few unique values, use mode; otherwise use median
                            non_null_values = df[col].dropna()
                            if len(non_null_values) > 0:  # If we have some non-null values
                                if non_null_values.nunique() < 10:
                                    # Use mode for columns with few unique values
                                    if not non_null_values.mode().empty:
                                        fill_val = non_null_values.mode()[0]
                                    else:
                                        fill_val = non_null_values.median() if len(non_null_values) > 0 else 0
                                else:
                                    # Use median for columns with many unique values
                                    fill_val = non_null_values.median() if len(non_null_values) > 0 else 0
                            else:
                                # No non-null values to derive statistics from
                                fill_val = 0
                            df[col] = df[col].fillna(fill_val)
                
                missing_after = df.isnull().sum().sum()
                st.info(f"Filled {missing_before - missing_after} missing values with most frequent values or appropriate substitutes")
            
            elif missing_handling == "Fill with zeros":
                df = df.fillna(0)
                st.info(f"Filled all missing values with zeros")
        
        # Show number of rows after handling missing values
        st.write(f"Final dataset: {df.shape[0]} rows, {df.shape[1]} columns")

        # Visualization options
        viz_col1, viz_col2 = st.columns(2)
        columns = df.columns.tolist()

        with viz_col1:
            highlighted_text("<h3>Data Preview</h3>")
            st.write(df.head())
            
            # Data summary
            if st.checkbox("Show data summary"):
                st.write("Dataset Shape:", df.shape)
                st.write("Column Types:")
                st.write(pd.DataFrame(df.dtypes, columns=["Data Type"]))
                
                numeric_df = df.select_dtypes(include=['float64', 'int64'])
                if not numeric_df.empty:
                    st.write("Numeric Data Summary:")
                    st.write(numeric_df.describe())
                
                # Show missing values count if any
                null_counts = df.isnull().sum()
                if null_counts.sum() > 0:
                    st.write("Missing Values Count:")
                    st.write(pd.DataFrame(null_counts, columns=["Count"]).query("Count > 0"))

        with viz_col2:
            highlighted_text("<h3>Select Visualization Options</h3>")
            x_axis = st.selectbox('Select the X-axis', options=columns + ["None"])
            y_axis = st.selectbox('Select the Y-axis', options=columns + ["None"])
            
            # Add optional color grouping
            add_color = st.checkbox("Add color grouping")
            if add_color:
                color_column = st.selectbox('Select column for color grouping', options=["None"] + columns)
            else:
                color_column = None
            
            plot_list = [
                "Line Plot", "Bar Chart", "Scatter Plot", "Distribution Plot",
                "Count Plot", "Correlation Heatmap", "Box Plot", "Violin Plot",
                "Hexbin Plot", "Pair Plot"
            ]
            plot_type = st.selectbox('Select the type of plot', options=plot_list)
            
            # Plot size options
            fig_width = st.slider("Figure Width", min_value=1, max_value=20, value=10)
            fig_height = st.slider("Figure Height", min_value=5, max_value=12, value=5)

        if st.button('Generate Plot'):
            highlighted_text(f"<h3>Visualization: {plot_type}</h3>")
            try:
                plt.style.use('dark_background')
                
                # Create figure with user-defined size
                fig = plt.figure(figsize=(fig_width, fig_height))

                if plot_type == "Line Plot" and x_axis != "None" and y_axis != "None":
                    if add_color and color_column != "None":
                        for category, group in df.groupby(color_column):
                            plt.plot(group[x_axis], group[y_axis], marker='o', linewidth=2, label=category)
                        plt.legend(title=color_column)
                    else:
                        plt.plot(df[x_axis], df[y_axis], color='#00BFFF', linewidth=2, marker='o')
                    plt.xlabel(x_axis)
                    plt.ylabel(y_axis)
                    plt.title('Line Plot')
                    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
                    st.pyplot(fig)

                elif plot_type == "Bar Chart" and x_axis != "None" and y_axis != "None":
                    if add_color and color_column != "None":
                        grouped = df.groupby([x_axis, color_column])[y_axis].mean().unstack()
                        grouped.plot(kind='bar', ax=plt.gca())
                        plt.legend(title=color_column)
                    else:
                        plt.bar(df[x_axis], df[y_axis], color='#00BFFF', alpha=0.8)
                    plt.xlabel(x_axis)
                    plt.ylabel(y_axis)
                    plt.title('Bar Chart')
                    plt.grid(axis='y', color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)

                elif plot_type == "Scatter Plot" and x_axis != "None" and y_axis != "None":
                    if add_color and color_column != "None":
                        for category, group in df.groupby(color_column):
                            plt.scatter(group[x_axis], group[y_axis], label=category, alpha=0.7)
                        plt.legend(title=color_column)
                    else:
                        plt.scatter(df[x_axis], df[y_axis], color='#00BFFF', alpha=0.7)
                    plt.xlabel(x_axis)
                    plt.ylabel(y_axis)
                    plt.title('Scatter Plot')
                    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
                    st.pyplot(fig)

                elif plot_type == "Distribution Plot" and y_axis != "None":
                    if add_color and color_column != "None":
                        for category, group in df.groupby(color_column):
                            sns.kdeplot(group[y_axis], label=category)
                        plt.legend(title=color_column)
                    else:
                        sns.histplot(df[y_axis], kde=True, color='#00BFFF')
                    plt.xlabel(y_axis)
                    plt.title('Distribution Plot')
                    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
                    st.pyplot(fig)

                elif plot_type == "Count Plot" and y_axis != "None":
                    if add_color and color_column != "None":
                        sns.countplot(x=df[y_axis], hue=df[color_column])
                    else:
                        sns.countplot(x=df[y_axis], palette=['#00BFFF', '#007BFF', '#0056B3'])
                    plt.xlabel(y_axis)
                    plt.title('Count Plot')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.grid(axis='y', color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
                    st.pyplot(fig)

                elif plot_type == "Correlation Heatmap":
                    # Get numeric columns only
                    numeric_df = df.select_dtypes(include=['float64', 'int64'])
                    
                    if not numeric_df.empty:
                        mask = np.triu(np.ones_like(numeric_df.corr(), dtype=bool))
                        sns.heatmap(numeric_df.corr(), annot=True, mask=mask,
                                    cmap='coolwarm', linewidths=0.5, fmt=".2f",
                                    annot_kws={"size": 8})
                        plt.title('Correlation Heatmap')
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        st.warning("No numeric columns available for correlation heatmap.")

                elif plot_type == "Box Plot" and y_axis != "None":
                    if x_axis != "None":
                        if add_color and color_column != "None":
                            sns.boxplot(data=df, x=x_axis, y=y_axis, hue=color_column)
                            plt.legend(title=color_column)
                        else:
                            sns.boxplot(data=df, x=x_axis, y=y_axis, palette=['#00BFFF', '#007BFF'])
                    else:
                        sns.boxplot(data=df, y=y_axis, color='#00BFFF')
                    plt.title('Box Plot')
                    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
                    plt.tight_layout()
                    st.pyplot(fig)

                elif plot_type == "Violin Plot" and y_axis != "None":
                    if x_axis != "None":
                        if add_color and color_column != "None":
                            sns.violinplot(data=df, x=x_axis, y=y_axis, hue=color_column, split=True)
                            plt.legend(title=color_column)
                        else:
                            sns.violinplot(data=df, x=x_axis, y=y_axis, palette=['#00BFFF', '#007BFF'])
                    else:
                        sns.violinplot(data=df, y=y_axis, color='#00BFFF')
                    plt.title('Violin Plot')
                    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
                    plt.tight_layout()
                    st.pyplot(fig)

                elif plot_type == "Hexbin Plot" and x_axis != "None" and y_axis != "None":
                    plt.hexbin(df[x_axis], df[y_axis], gridsize=30, cmap='Blues')
                    plt.colorbar(label='Density')
                    plt.xlabel(x_axis)
                    plt.ylabel(y_axis)
                    plt.title('Hexbin Plot')
                    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
                    st.pyplot(fig)
                    
                elif plot_type == "Pair Plot":
                    # Get numeric columns only (limit to first 5 to avoid cluttering)
                    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()[:5]
                    
                    if len(numeric_cols) > 1:
                        # Close the current figure
                        plt.close(fig)
                        
                        # Create a new pairplot
                        if add_color and color_column != "None":
                            pair_plot = sns.pairplot(df[numeric_cols + [color_column]], hue=color_column, height=2.5)
                        else:
                            pair_plot = sns.pairplot(df[numeric_cols], height=2.5)
                            
                        st.pyplot(pair_plot.fig)
                        
                        # Reassign fig for download purposes
                        fig = pair_plot.fig
                    else:
                        st.warning("Need at least two numeric columns for a pair plot.")
                
                else:
                    st.warning("Please select appropriate X and Y axes for the chosen plot type.")
                    
                # Create download link for the plot
                if 'fig' in locals():
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    download_formats = st.columns(3)
                    with download_formats[0]:
                        st.markdown(get_image_download_link(fig, f"{plot_type}_{timestamp}", "png"), unsafe_allow_html=True)
                    with download_formats[1]:
                        st.markdown(get_image_download_link(fig, f"{plot_type}_{timestamp}", "svg"), unsafe_allow_html=True)
                    with download_formats[2]:
                        st.markdown(get_image_download_link(fig, f"{plot_type}_{timestamp}", "pdf"), unsafe_allow_html=True)
                    
                    # Option to download data as CSV
                    csv = df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="dataset_{timestamp}.csv" class="download-btn">Download Dataset as CSV</a>'
                    st.markdown(href, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"An error occurred while generating the plot: {e}")

    except Exception as e:
        st.error(f"Error reading the CSV file: {e}")
else:
    st.markdown("""
    <div class="highlight-container">
    <h3>Welcome to the Data Visualizer!</h3>
    <p>Upload one or two CSV files to start generating visualizations. This tool supports:</p>
    <ul>
        <li>Automatic merging or concatenation of two datasets</li>
        <li>Intelligent missing value handling</li>
        <li>Line plots</li>
        <li>Bar charts</li>
        <li>Scatter plots</li>
        <li>Distribution plots</li>
        <li>Count plots</li>
        <li>Correlation heatmaps</li>
        <li>Box plots</li>
        <li>Violin plots</li>
        <li>Hexbin plots</li>
        <li>Pair plots</li>
        <li>Download plots in PNG, SVG, or PDF formats</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)