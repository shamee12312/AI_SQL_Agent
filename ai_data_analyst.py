import streamlit as st
import pandas as pd
import numpy as np
import duckdb
import json
import io
from typing import Dict, Any, Optional, Tuple
import traceback
from datetime import datetime
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Configure page
st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

class SchemaAnalyzer:
    """Analyzes and extracts schema information from uploaded data files"""
    
    @staticmethod
    def analyze_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
        """Extract comprehensive schema information from DataFrame"""
        schema_info = {
            'table_name': 'data',
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'columns': {},
            'sample_data': {},
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        for col in df.columns:
            col_info = {
                'dtype': str(df[col].dtype),
                'null_count': df[col].isnull().sum(),
                'null_percentage': (df[col].isnull().sum() / len(df)) * 100,
                'unique_count': df[col].nunique()
            }
            
            # Type-specific analysis
            if df[col].dtype == 'object':
                # String/categorical analysis
                unique_vals = df[col].dropna().unique()
                if len(unique_vals) <= 20:  # Categorical threshold
                    col_info['categorical_values'] = unique_vals.tolist()
                else:
                    col_info['sample_values'] = unique_vals[:10].tolist()
                col_info['avg_string_length'] = df[col].astype(str).str.len().mean()
                
            elif df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                # Numeric analysis
                col_info['min_value'] = df[col].min()
                col_info['max_value'] = df[col].max()
                col_info['mean_value'] = df[col].mean()
                col_info['std_dev'] = df[col].std()
                col_info['median_value'] = df[col].median()
                
            elif df[col].dtype == 'datetime64[ns]':
                # Date analysis
                col_info['min_date'] = str(df[col].min())
                col_info['max_date'] = str(df[col].max())
                col_info['date_range_days'] = (df[col].max() - df[col].min()).days
            
            schema_info['columns'][col] = col_info
            
            # Sample data for context
            sample_vals = df[col].dropna().head(3).tolist()
            schema_info['sample_data'][col] = sample_vals
            
        return schema_info

class GeminiQueryGenerator:
    """Handles natural language to SQL conversion using Gemini API"""
    
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-pro')
        
    def generate_sql_query(self, user_prompt: str, schema_info: Dict[str, Any]) -> Tuple[str, str]:
        """Generate SQL query from natural language prompt"""
        
        # Create context-optimized schema description
        schema_context = self._create_schema_context(schema_info)
        
        system_prompt = f"""You are an expert SQL query generator. Convert natural language questions into valid SQL queries.

DATABASE SCHEMA:
{schema_context}

IMPORTANT RULES:
1. Always use 'data' as the table name
2. Generate only SELECT statements (no INSERT, UPDATE, DELETE, DROP)
3. Use proper SQL syntax compatible with DuckDB
4. Include appropriate WHERE, GROUP BY, ORDER BY clauses as needed
5. For aggregations, always include proper GROUP BY clauses
6. Use column names exactly as provided in schema
7. Return only the SQL query without explanations
8. If the question is unclear, make reasonable assumptions based on available columns

USER QUESTION: {user_prompt}

Generate the SQL query:"""

        try:
            response = self.model.generate_content(
                system_prompt,
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
            )
            
            sql_query = response.text.strip()
            # Clean up the response
            if sql_query.startswith('```'):
                sql_query = sql_query[6:]
            if sql_query.endswith('```'):
                sql_query = sql_query[:-3]
            sql_query = sql_query.strip()
            
            return sql_query, "success"
            
        except Exception as e:
            return f"Error generating query: {str(e)}", "error"
    
    def _create_schema_context(self, schema_info: Dict[str, Any]) -> str:
        """Create optimized schema context for AI"""
        context_parts = []
        
        context_parts.append(f"Table: {schema_info['table_name']}")
        context_parts.append(f"Total Rows: {schema_info['total_rows']:,}")
        context_parts.append(f"Total Columns: {schema_info['total_columns']}")
        context_parts.append("\nCOLUMNS:")
        
        for col_name, col_info in schema_info['columns'].items():
            col_desc = f"  - {col_name} ({col_info['dtype']})"
            
            if col_info['dtype'] == 'object':
                if 'categorical_values' in col_info:
                    col_desc += f" - Values: {col_info['categorical_values']}"
                elif 'sample_values' in col_info:
                    col_desc += f" - Sample: {col_info['sample_values']}"
            elif col_info['dtype'] in ['int64', 'float64', 'int32', 'float32']:
                col_desc += f" - Range: {col_info['min_value']} to {col_info['max_value']}"
            
            if col_info['null_count'] > 0:
                col_desc += f" - Nulls: {col_info['null_count']}"
                
            context_parts.append(col_desc)
        
        return "\n".join(context_parts)
    
    def explain_results(self, query: str, results: pd.DataFrame, user_prompt: str) -> str:
        """Convert query results back to natural language"""
        
        # Prepare results summary
        results_summary = self._prepare_results_summary(results)
        
        explain_prompt = f"""You are a data analyst explaining query results in natural language.

ORIGINAL QUESTION: {user_prompt}
SQL QUERY EXECUTED: {query}

RESULTS SUMMARY:
{results_summary}

Provide a clear, concise explanation of what the data shows in response to the original question. 
Focus on insights and key findings. Use natural language and avoid technical jargon.
If there are specific numbers, highlight the most important ones.
"""

        try:
            response = self.model.generate_content(explain_prompt)
            return response.text
        except Exception as e:
            return f"Results retrieved successfully, but explanation failed: {str(e)}"
    
    def _prepare_results_summary(self, df: pd.DataFrame) -> str:
        """Prepare a summary of results for AI explanation"""
        if df.empty:
            return "No results found."
        
        summary_parts = []
        summary_parts.append(f"Number of rows: {len(df)}")
        summary_parts.append(f"Columns: {list(df.columns)}")
        
        # Add sample data (first few rows)
        if len(df) > 0:
            summary_parts.append("\nSample data:")
            sample_size = min(5, len(df))
            for i in range(sample_size):
                row_data = {}
                for col in df.columns:
                    row_data[col] = df.iloc[i][col]
                summary_parts.append(f"Row {i+1}: {row_data}")
        
        # Add basic statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            summary_parts.append("\nNumeric column statistics:")
            for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                summary_parts.append(f"{col}: min={df[col].min()}, max={df[col].max()}, avg={df[col].mean():.2f}")
        
        return "\n".join(summary_parts)

class DataQueryEngine:
    """Executes SQL queries on pandas DataFrames using DuckDB"""
    
    @staticmethod
    def execute_query(df: pd.DataFrame, query: str) -> Tuple[pd.DataFrame, str, str]:
        """Execute SQL query and return results"""
        try:
            # Create DuckDB connection
            conn = duckdb.connect()
            
            # Register DataFrame as a table
            conn.register('data', df)
            
            # Execute query
            result_df = conn.execute(query).fetch_df()
            
            # Close connection
            conn.close()
            
            return result_df, "success", "Query executed successfully"
            
        except Exception as e:
            error_msg = f"Query execution failed: {str(e)}"
            return pd.DataFrame(), "error", error_msg

def main():
    """Main Streamlit application"""
    
    st.title("🤖 AI-Powered Data Analyst")
    st.markdown("Upload your data, ask questions in natural language, and get instant insights!")
    
    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")
    
    # API Key input
    api_key = st.sidebar.text_input(
        "Google Gemini API Key",
        type="password",
        help="Get your API key from https://makersuite.google.com/app/apikey"
    )
    
    if not api_key:
        st.sidebar.warning("⚠️ Please enter your Gemini API key to continue")
        st.info("Please enter your Google Gemini API key in the sidebar to get started.")
        return
    
    # Initialize components
    try:
        query_generator = GeminiQueryGenerator(api_key)
        query_engine = DataQueryEngine()
        schema_analyzer = SchemaAnalyzer()
        st.sidebar.success("✅ API connected successfully")
    except Exception as e:
        st.sidebar.error(f"❌ API connection failed: {str(e)}")
        return
    
    # File upload section
    st.header("📁 Data Upload")
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload your data file to analyze"
    )
    
    if uploaded_file is not None:
        # Load data
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
                
            st.success(f"✅ File loaded successfully: {len(df)} rows, {len(df.columns)} columns")
            
            # Store data in session state
            st.session_state['df'] = df
            st.session_state['uploaded_file_name'] = uploaded_file.name
            
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            return
    
    # Show data preview and schema
    if 'df' in st.session_state:
        df = st.session_state['df']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
        
        with col2:
            st.subheader("🔍 Schema Information")
            
            # Analyze schema
            if 'schema_info' not in st.session_state:
                with st.spinner("Analyzing schema..."):
                    schema_info = schema_analyzer.analyze_dataframe(df)
                    st.session_state['schema_info'] = schema_info
            
            schema_info = st.session_state['schema_info']
            
            # Display schema info
            st.metric("Total Rows", f"{schema_info['total_rows']:,}")
            st.metric("Total Columns", schema_info['total_columns'])
            st.metric("Memory Usage", f"{schema_info['memory_usage_mb']:.1f} MB")
            
            # Column information
            with st.expander("Column Details"):
                for col_name, col_info in schema_info['columns'].items():
                    st.write(f"**{col_name}** ({col_info['dtype']})")
                    if col_info['null_count'] > 0:
                        st.write(f"  - Null values: {col_info['null_count']} ({col_info['null_percentage']:.1f}%)")
                    st.write(f"  - Unique values: {col_info['unique_count']}")
        
        # Query section
        st.header("💬 Ask Questions About Your Data")
        
        # Predefined example questions
        st.subheader("💡 Example Questions")
        example_questions = [
            "Show me the first 10 rows",
            "What are the column names and their data types?",
            "Show summary statistics for all numeric columns",
            "Which columns have missing values?",
            "Group by the first categorical column and count rows"
        ]
        
        cols = st.columns(len(example_questions))
        for i, question in enumerate(example_questions):
            if cols[i].button(question, key=f"example_{i}"):
                st.session_state['user_query'] = question
        
        # User input
        user_query = st.text_area(
            "Enter your question:",
            value=st.session_state.get('user_query', ''),
            height=100,
            placeholder="e.g., 'Show me the top 5 rows sorted by the first numeric column' or 'What is the average value of column X?'"
        )
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        if col1.button("🔍 Analyze", type="primary"):
            if user_query.strip():
                analyze_data(user_query, df, schema_info, query_generator, query_engine)
            else:
                st.warning("Please enter a question about your data.")
        
        if col2.button("🔄 Clear Results"):
            if 'results_history' in st.session_state:
                del st.session_state['results_history']
            st.rerun()
        
        # Show analysis history
        if 'results_history' in st.session_state:
            st.header("📋 Analysis History")
            for i, result in enumerate(reversed(st.session_state['results_history'])):
                with st.expander(f"Q: {result['question'][:80]}{'...' if len(result['question']) > 80 else ''}"):
                    st.write("**Question:**", result['question'])
                    st.write("**Generated SQL:**")
                    st.code(result['sql_query'], language='sql')
                    
                    if result['status'] == 'success':
                        st.write("**Results:**")
                        st.dataframe(result['results'], use_container_width=True)
                        st.write("**AI Explanation:**")
                        st.write(result['explanation'])
                    else:
                        st.error(f"Error: {result['error']}")

def analyze_data(user_query: str, df: pd.DataFrame, schema_info: Dict, query_generator: GeminiQueryGenerator, query_engine: DataQueryEngine):
    """Perform data analysis based on user query"""
    
    with st.spinner("🤖 Generating SQL query..."):
        # Generate SQL query
        sql_query, gen_status = query_generator.generate_sql_query(user_query, schema_info)
        
        if gen_status != "success":
            st.error(f"Failed to generate query: {sql_query}")
            return
    
    st.subheader("🔧 Generated SQL Query")
    st.code(sql_query, language='sql')
    
    with st.spinner("⚡ Executing query..."):
        # Execute query
        results_df, exec_status, exec_message = query_engine.execute_query(df, sql_query)
        
        if exec_status != "success":
            st.error(f"Query execution failed: {exec_message}")
            return
    
    st.subheader("📊 Query Results")
    
    if len(results_df) > 0:
        st.dataframe(results_df, use_container_width=True)
        
        # Download results
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="💾 Download Results as CSV",
            data=csv,
            file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("Query executed successfully but returned no results.")
    
    with st.spinner("🧠 Generating explanation..."):
        # Generate natural language explanation
        explanation = query_generator.explain_results(sql_query, results_df, user_query)
    
    st.subheader("💡 AI Explanation")
    st.write(explanation)
    
    # Store in history
    if 'results_history' not in st.session_state:
        st.session_state['results_history'] = []
    
    result_entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'question': user_query,
        'sql_query': sql_query,
        'results': results_df,
        'explanation': explanation,
        'status': 'success'
    }
    
    st.session_state['results_history'].append(result_entry)

if __name__ == "__main__":
    main()
