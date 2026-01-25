import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
import base64

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
            position: relative;
        }
        .download-btn {
        position: absolute;
        top: 10px;
        right: 10px;
        background-color: black;
        color: black;  /* Changed from white to black for better visibility */
        border: none;
        padding: 5px 10px;
        border-radius: 4px;
        cursor: pointer;
        font-size: 14px;
        font-weight: bold;  /* Added bold for better visibility */
        z-index: 100;
    }
    .download-btn:hover {
        background-color:black;
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
        .suggestion-card {
            background-color: rgba(0, 191, 255, 0.15);
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 10px;
            border-left: 3px solid #00BFFF;
        }
        .suggestion-title {
            font-weight: bold;
            color: #00BFFF;
            margin-bottom: 5px;
        }
        .suggestion-reason {
            font-size: 0.9em;
            opacity: 0.9;
        }
        .suggestion-button {
            background-color: #00BFFF !important;
            color: white !important;
            font-weight: bold !important;
            border-radius: 20px !important;
            padding: 5px 15px !important;
            margin-top: 10px !important;
            border: none !important;
            box-shadow: 0px 3px 6px rgba(0, 0, 0, 0.2) !important;
        }
        .magic-btn {
            background: linear-gradient(45deg, #00BFFF, #007BFF) !important;
            color: white !important;
            font-weight: bold !important;
            border-radius: 20px !important;
            padding: 10px 20px !important;
            margin: 15px 0 !important;
            border: none !important;
            box-shadow: 0px 3px 6px rgba(0, 0, 0, 0.3) !important;
            transition: all 0.3s ease !important;
        }
        .magic-btn:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0px 5px 10px rgba(0, 0, 0, 0.4) !important;
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

# Function to create download link for plot
def get_image_download_link(fig, filename="plot.png", text="Download Plot"):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}" class="download-btn">{text} 📥</a>'
    return href

# Container for plots with download button
def plot_container(plot_function):
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    fig = plot_function()
    if fig:
        # Add download button
        download_link = get_image_download_link(fig)
        st.markdown(download_link, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

color_options = {
    "Sky Blue": "#00BFFF",
    "Crimson": "#DC143C",
    "Lime Green": "#32CD32",
    "Orange": "#FFA500",
    "Purple": "#800080",
    "Black": "#000000",
    "Gold": "#FFD700"
}

# New function to analyze data and suggest appropriate plots
def suggest_plots(df):
    suggestions = []
    column_types = {}
    
    # Identify column types
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].nunique() <= 10:
                column_types[col] = "categorical_numeric"
            else:
                column_types[col] = "continuous_numeric"
        elif pd.api.types.is_categorical_dtype(df[col]) or df[col].nunique() <= 20:
            column_types[col] = "categorical"
        else:
            column_types[col] = "text"
    
    # Get categorical and numerical columns
    categorical_cols = [col for col, type_ in column_types.items() 
                      if type_ in ["categorical", "categorical_numeric"]]
    numeric_cols = [col for col, type_ in column_types.items() 
                   if type_ in ["continuous_numeric", "categorical_numeric"]]
    
    # Suggestion 1: Distribution analysis for numeric columns
    if numeric_cols:
        for col in numeric_cols[:3]:  # Limit to first 3 columns to avoid overwhelming
            suggestions.append({
                "plot_type": "Distribution Plot",
                "x_axis": "None",
                "y_axis": col,
                "title": f"Distribution of {col}",
                "reason": f"Understand the distribution pattern of {col} values."
            })
    
    # Suggestion 2: Correlation heatmap if multiple numeric columns
    if len(numeric_cols) >= 2:
        suggestions.append({
            "plot_type": "Correlation Heatmap",
            "x_axis": "None",
            "y_axis": "None",
            "title": "Correlation Heatmap",
            "reason": f"Explore relationships between the {len(numeric_cols)} numeric variables."
        })
    
    # Suggestion 3: Scatter plots for pairs of numeric columns
    if len(numeric_cols) >= 2:
        for i in range(min(3, len(numeric_cols))):
            for j in range(i+1, min(4, len(numeric_cols))):
                suggestions.append({
                    "plot_type": "Scatter Plot",
                    "x_axis": numeric_cols[i],
                    "y_axis": numeric_cols[j],
                    "title": f"{numeric_cols[i]} vs {numeric_cols[j]}",
                    "reason": f"Explore relationship between {numeric_cols[i]} and {numeric_cols[j]}."
                })
    
    # Suggestion 4: Count plots for categorical columns
    if categorical_cols:
        for col in categorical_cols[:3]:  # Limit to first 3 columns
            suggestions.append({
                "plot_type": "Count Plot",
                "x_axis": "None",
                "y_axis": col,
                "title": f"Count of {col}",
                "reason": f"See the frequency distribution of {col} categories."
            })
    
    # Suggestion 5: Box plots for numeric by categorical
    if numeric_cols and categorical_cols:
        for num_col in numeric_cols[:2]:
            for cat_col in categorical_cols[:2]:
                if df[cat_col].nunique() <= 10:  # Limit to categories that won't overcrowd the plot
                    suggestions.append({
                        "plot_type": "Box Plot",
                        "x_axis": cat_col,
                        "y_axis": num_col,
                        "title": f"{num_col} by {cat_col}",
                        "reason": f"Compare distribution of {num_col} across different {cat_col} categories."
                    })
    
    # Suggestion 6: Time series if date column detected
    date_cols = []
    for col in df.columns:
        try:
            if pd.to_datetime(df[col], errors='coerce').notna().all():
                date_cols.append(col)
        except:
            pass
    
    if date_cols and numeric_cols:
        for date_col in date_cols[:1]:
            for num_col in numeric_cols[:2]:
                suggestions.append({
                    "plot_type": "Line Plot",
                    "x_axis": date_col,
                    "y_axis": num_col,
                    "title": f"{num_col} over Time",
                    "reason": f"Track how {num_col} changes over time."
                })
    
    # Suggestion 7: Bar charts for categorical vs numeric
    if categorical_cols and numeric_cols:
        for cat_col in categorical_cols[:2]:
            for num_col in numeric_cols[:2]:
                if df[cat_col].nunique() <= 15:  # Limit to avoid overcrowded bar charts
                    suggestions.append({
                        "plot_type": "Bar Chart",
                        "x_axis": cat_col,
                        "y_axis": num_col,
                        "title": f"{num_col} by {cat_col}",
                        "reason": f"Compare {num_col} values across different {cat_col} categories."
                    })
    
    return suggestions

# Function to generate plots based on given parameters
def generate_plot(df, plot_type, x_axis, y_axis, color, fig_width=10, fig_height=6):
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    if plot_type == "Line Plot" and x_axis != "None" and y_axis != "None":
        plt.plot(df[x_axis], df[y_axis], color=color, linewidth=2, marker='o')
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.title('Line Plot')
        plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        
    elif plot_type == "Bar Chart" and x_axis != "None" and y_axis != "None":
        plt.bar(df[x_axis], df[y_axis], color=color, alpha=0.8)
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.title('Bar Chart')
        plt.grid(axis='y', color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
    elif plot_type == "Scatter Plot" and x_axis != "None" and y_axis != "None":
        plt.scatter(df[x_axis], df[y_axis], color=color, alpha=0.7)
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.title('Scatter Plot')
        plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        
    elif plot_type == "Distribution Plot" and y_axis != "None":
        sns.histplot(df[y_axis], kde=True, color=color)
        plt.xlabel(y_axis)
        plt.title('Distribution Plot')
        plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        
    elif plot_type == "Count Plot" and y_axis != "None":
        sns.countplot(x=df[y_axis], palette=['#00BFFF', '#007BFF', '#0056B3'])
        plt.xlabel(y_axis)
        plt.title('Count Plot')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.grid(axis='y', color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        
    elif plot_type == "Correlation Heatmap":
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        if not numeric_df.empty:
            fig = plt.figure(figsize=(fig_width, fig_height))
            mask = np.triu(np.ones_like(numeric_df.corr(), dtype=bool))
            sns.heatmap(numeric_df.corr(), annot=True, mask=mask,
                        cmap='coolwarm', linewidths=0.5, fmt=".2f",
                        annot_kws={"size": 8})
            plt.title('Correlation Heatmap')
            plt.tight_layout()
            return fig
        else:
            st.warning("No numeric columns available for correlation heatmap.")
            return None
            
    elif plot_type == "Box Plot" and y_axis != "None":
        if x_axis != "None":
            sns.boxplot(data=df, x=x_axis, y=y_axis, color=color)
        else:
            sns.boxplot(data=df, y=y_axis, color='#00BFFF')
        plt.title('Box Plot')
        plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.tight_layout()
        
    elif plot_type == "Violin Plot" and y_axis != "None":
        if x_axis != "None":
            sns.violinplot(data=df, x=x_axis, y=y_axis, palette=['#00BFFF', '#007BFF'])
        else:
            sns.violinplot(data=df, y=y_axis, color='#00BFFF')
        plt.title('Violin Plot')
        plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.tight_layout()
        
    elif plot_type == "Hexbin Plot" and x_axis != "None" and y_axis != "None":
        plt.hexbin(df[x_axis], df[y_axis], gridsize=30, cmap='Blues')
        plt.colorbar(label='Density')
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.title('Hexbin Plot')
        plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        
    else:
        st.warning("Please select appropriate X and Y axes for the chosen plot type.")
        return None
    
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)    
    return fig

# Apply style functions
add_bg_from_url()
add_overlay()
inject_custom_css()

st.title('⚡ Data Plotter: Instant Plot Generator')

highlighted_text("<h3>Upload your CSV file</h3><p>Select a CSV file to analyze and generate visualizations</p>")

uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "json"])

def load_data(file):
    ext = file.name.split('.')[-1]
    if ext == 'csv':
        return pd.read_csv(file)
    elif ext == 'xlsx':
        return pd.read_excel(file)
    elif ext == 'json':
        return pd.read_json(file)
    else:
        return None

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        col1, col2 = st.columns(2)
        columns = df.columns.tolist()

        with col1:
            highlighted_text("<h3>Data Preview</h3>")
            st.write(df.head())

            # Show shape
            st.success(f"📦 Data Shape: {df.shape[0]} rows × {df.shape[1]} columns")

            # Optional data summary
            if st.checkbox("🔍 Show Data Types Summary"):
                st.write(df.dtypes)
                
            # Add new Auto Plot Suggestion button with magic effect
            st.markdown("""
                <div class="highlight-container">
                <h3>✨ AI Plot Recommendation</h3>
                <p>Let AI analyze your data and suggest the best visualizations</p>
                </div>
                """, unsafe_allow_html=True)
                
            show_suggestions = st.button("✨ Suggest Plots For Me", key="suggest_plots", 
                                         help="Analyze data and suggest appropriate visualizations")

        with col2:
            highlighted_text("<h3>Select Visualization Options</h3>")
            x_axis = st.selectbox('Select the X-axis', options=columns + ["None"])
            y_axis = st.selectbox('Select the Y-axis', options=columns + ["None"])

            plot_list = [
                "Line Plot", "Bar Chart", "Scatter Plot", "Distribution Plot",
                "Count Plot", "Correlation Heatmap", "Box Plot", "Violin Plot",
                "Hexbin Plot"
            ]
            
            plot_type = st.selectbox('Select the type of plot', options=plot_list)
            selected_color = st.selectbox("🎨 Choose Plot Color", options=list(color_options.keys()))
        
        # Common plot settings
        fig_width = st.slider("Figure Width", min_value=1, max_value=20, value=10)
        fig_height = st.slider("Figure Height", min_value=1, max_value=12, value=5)

        # Display plot suggestions if requested
        if show_suggestions:
            st.markdown("""
                <div class="highlight-container">
                <h3>🔍 AI Plot Recommendations</h3>
                <p>Based on your data characteristics, here are the recommended visualizations:</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Get plot suggestions
            suggestions = suggest_plots(df)
            
            if suggestions:
                for i, suggestion in enumerate(suggestions):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"""
                            <div class="suggestion-card">
                                <div class="suggestion-title">{suggestion['title']}</div>
                                <div class="suggestion-reason">{suggestion['reason']}</div>
                                <div>Plot Type: {suggestion['plot_type']}</div>
                                <div>X-axis: {suggestion['x_axis']}</div>
                                <div>Y-axis: {suggestion['y_axis']}</div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        if st.button(f"Apply", key=f"apply_{i}"):
                            plot_type = suggestion['plot_type']
                            x_axis = suggestion['x_axis']
                            y_axis = suggestion['y_axis']
                            
                            try:
                                color = color_options[selected_color]
                                fig = generate_plot(df, plot_type, x_axis, y_axis, color, fig_width, fig_height)
                                
                                if fig:
                                    st.pyplot(fig)
                                    download_link = get_image_download_link(fig, f"{plot_type.lower().replace(' ', '_')}.png", "Download Plot")
                                    st.markdown(download_link, unsafe_allow_html=True)
                            except Exception as e:
                                st.error(f"Error generating suggested plot: {e}")
            else:
                st.info("No plot suggestions could be generated for this dataset.")

        # Manual plot generation
        if st.button('Generate Plot'):
            highlighted_text(f"<h3>Visualization: {plot_type}</h3>")
            try:
                color = color_options[selected_color]
                fig = generate_plot(df, plot_type, x_axis, y_axis, color, fig_width, fig_height)
                
                if fig:
                    # Display the plot
                    st.pyplot(fig)
                    
                    # Add download button
                    download_link = get_image_download_link(fig, f"{plot_type.lower().replace(' ', '_')}.png", "Download Plot")
                    st.markdown(download_link, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"An error occurred while generating the plot: {e}")

    except Exception as e:
        st.error(f"Error reading the CSV file: {e}")
    except pd.errors.EmptyDataError:
        st.error("The file is empty. Please upload a CSV file with data.")
    except pd.errors.ParserError:
        st.error("There was an error parsing the file. Please ensure it is a valid csv format.")
else:
    st.markdown("""
    <div class="highlight-container">
    <h3>Welcome to the Data Visualizer!</h3>
    <p>Upload a CSV file to start generating visualizations. This tool supports:</p>
    <ul>
        <li>Line plots</li>
        <li>Bar charts</li>
        <li>Scatter plots</li>
        <li>Distribution plots</li>
        <li>Count plots</li>
        <li>Correlation heatmaps</li>
        <li>Box plots</li>
        <li>Violin plots</li>
        <li>Hexbin plots</li>
    </ul>
    <p><strong>NEW!</strong> ✨ AI Plot Recommendation - Let the app analyze your data and suggest the best visualizations!</p>
    </div>
    """, unsafe_allow_html=True)