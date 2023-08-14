import pandas as pd
import streamlit as st
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid import GridUpdateMode, DataReturnMode
import warnings


# hide future warnings (caused by st_aggrid)
warnings.simplefilter(action='ignore', category=FutureWarning)

#set page layout and define basic variables
st.set_page_config(layout="wide", page_icon='?', page_title="TBD Cool Name")


# Session states
# --------------


   
@st.cache_data
def get_data():
    """Returns a pandas DataFrame."""
    df = pd.read_parquet("legal_reddit_df_with_embeddings.parquet")
    return df


df = get_data()

# select columns to show
df_filtered = df[['id', 'title', 'body', 'text_label', 'full_link']]


# apply proper capitalization to column names and replace underscore with space
# df_filtered.columns = df_filtered.columns.str.title().str.replace('_', ' ')

# creating AgGrid dynamic table and setting configurations
gb = GridOptionsBuilder.from_dataframe(df_filtered)
gb.configure_selection(selection_mode="single", use_checkbox=True)
gb.configure_column(field='id', width=40)
gb.configure_column(field='title', width=260)
gb.configure_column(field='body', width=350)
gb.configure_column(field='text_label', width=60)
gb.configure_column(field='full_link', width=240)

gridOptions = gb.build()

response = AgGrid(
    df_filtered,
    gridOptions=gridOptions,
    enable_enterprise_modules=False,
    height=600,
    update_mode=GridUpdateMode.SELECTION_CHANGED,
    data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
    fit_columns_on_grid_load=False,
    theme='alpine',
    allow_unsafe_jscode=True
)


