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
        fig_width = st.slider("Figure Width", min_value=1, max_value=20, value=10)
        fig_height = st.slider("Figure Height", min_value=1, max_value=12, value=5)

        if st.button('Generate Plot'):
            highlighted_text(f"<h3>Visualization: {plot_type}</h3>")
            try:
                plt.style.use('dark_background')
                color = color_options[selected_color]

                def generate_plot():
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
                        plt.scatter(df[x_axis], df[y_axis], color=color , alpha=0.7)
                        plt.xlabel(x_axis)
                        plt.ylabel(y_axis)
                        plt.title('Scatter Plot')
                        plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
                        
                    elif plot_type == "Distribution Plot" and y_axis != "None":
                        sns.histplot(df[y_axis], kde=True, color=color)
                        plt.xlabel(y_axis)
                        plt.ylabel("Count")
                        plt.title(f'Distribution plot of {y_axis}')
        
                        plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
                        
                    elif plot_type == "Count Plot" and y_axis != "None":
                        sns.countplot(x=df[y_axis], palette=['#00BFFF', '#007BFF', '#0056B3'])
                        plt.xlabel(y_axis)
                        plt.ylabel("Count")
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

                # Create plot container with plot and download button
                fig = generate_plot()
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
    </div>
    """, unsafe_allow_html=True)