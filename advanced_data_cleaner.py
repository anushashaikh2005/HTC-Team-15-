"""
Advanced Data Cleaner - Enterprise-grade CSV data cleaning system
"""

import os
import sys
import time
import logging
import argparse
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Tuple, Any, Optional, Callable
from dataclasses import dataclass, field
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import re
import yaml
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import warnings
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from scipy import stats
import chardet
import hashlib
from tqdm import tqdm
import joblib

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('data_cleaner.log')
    ]
)

logger = logging.getLogger('AdvancedDataCleaner')

#################################################
# Configuration Classes
#################################################

@dataclass
class MissingValueConfig:
    """Configuration for missing value handling"""
    strategy: str = 'auto'  # 'auto', 'drop', 'impute', 'flag', 'none'
    imputation_method: str = 'auto'  # 'mean', 'median', 'mode', 'knn', 'iterative', 'constant', 'auto'
    threshold_row: float = 0.5  # Drop rows with more than this fraction of missing values
    threshold_column: float = 0.5  # Drop columns with more than this fraction of missing values
    fill_value: Union[str, int, float] = 0  # Value to use when imputation_method is 'constant'
    knn_neighbors: int = 5  # Number of neighbors for KNN imputation
    add_indicator: bool = True  # Whether to add indicator columns for missing values
    group_columns: List[str] = field(default_factory=list)  # For grouped imputation


@dataclass
class OutlierConfig:
    """Configuration for outlier detection and handling"""
    detection_method: str = 'auto'  # 'zscore', 'iqr', 'isolation_forest', 'none', 'auto'
    threshold: float = 3.0  # For Z-score method
    iqr_multiplier: float = 1.5  # For IQR method
    contamination: float = 0.05  # For isolation forest
    handling_strategy: str = 'auto'  # 'remove', 'cap', 'flag', 'none', 'auto'
    columns_to_check: List[str] = field(default_factory=list)  # Empty means all numeric


@dataclass
class DataTypeConfig:
    """Configuration for data type handling"""
    auto_convert: bool = True  # Automatically convert data types
    date_format: str = 'auto'  # Format string for date parsing
    infer_objects: bool = True  # Use pandas infer_objects
    convert_numeric: bool = True  # Try to convert strings to numeric
    true_values: List[str] = field(default_factory=lambda: ['yes', 'true', 'y', 't', '1'])
    false_values: List[str] = field(default_factory=lambda: ['no', 'false', 'n', 'f', '0'])
    type_mappings: Dict[str, str] = field(default_factory=dict)  # Manual column type mappings


@dataclass
class DuplicateConfig:
    """Configuration for duplicate handling"""
    strategy: str = 'auto'  # 'remove_first', 'remove_last', 'remove_all', 'flag', 'none', 'auto'
    subset: List[str] = field(default_factory=list)  # Columns to consider, empty means all
    keep: str = 'first'  # 'first', 'last', False
    ignore_case: bool = False  # Whether to ignore case in string comparisons


@dataclass
class NormalizationConfig:
    """Configuration for data normalization"""
    method: str = 'none'  # 'standardize', 'minmax', 'robust', 'log', 'none'
    columns: List[str] = field(default_factory=list)  # Empty means all numeric
    custom_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)


@dataclass
class EncodingConfig:
    """Configuration for categorical encoding"""
    method: str = 'auto'  # 'one_hot', 'label', 'binary', 'frequency', 'target', 'none', 'auto'
    columns: List[str] = field(default_factory=list)  # Empty means all categorical
    max_categories: int = 10  # Max unique values for auto one-hot encoding
    drop_first: bool = False  # For one-hot encoding
    target_column: str = ''  # For target encoding


@dataclass
class ValidationConfig:
    """Configuration for data validation"""
    enabled: bool = True
    rules: Dict[str, Dict] = field(default_factory=dict)
    action_on_fail: str = 'flag'  # 'drop', 'flag', 'fix', 'error'


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization"""
    chunk_size: int = 100000  # For processing large files in chunks
    n_jobs: int = -1  # Number of parallel jobs, -1 means all cores
    low_memory: bool = False  # Use pandas low_memory option
    optimize_dtypes: bool = True  # Optimize pandas dtypes for memory usage


@dataclass
class OutputConfig:
    """Configuration for output options"""
    cleaned_suffix: str = '_cleaned'
    create_report: bool = True
    report_format: str = 'html'  # 'html', 'json', 'txt'
    save_intermediate: bool = False
    compress_output: bool = False
    output_format: str = 'csv'  # 'csv', 'parquet', 'excel', 'json', etc.
    index: bool = False  # Whether to include index in output
    verbose: int = 1  # 0 = quiet, 1 = normal, 2 = debug


@dataclass
class CleanerConfig:
    """Master configuration for the data cleaner"""
    input_file: str = ''
    output_file: str = ''
    delimiter: str = 'auto'  # 'auto', ',', '\t', '|', etc.
    encoding: str = 'auto'  # 'auto', 'utf-8', 'latin1', etc.
    skip_rows: int = 0
    skip_footer: int = 0
    header: Union[int, List[int], None] = 0
    na_values: List[str] = field(default_factory=lambda: ['NA', 'N/A', '', 'null', 'NULL', 'nan', 'NaN', 'None'])
    true_values: List[str] = field(default_factory=lambda: ['yes', 'true', 'y', 't', '1'])
    false_values: List[str] = field(default_factory=lambda: ['no', 'false', 'n', 'f', '0'])
    comment: str = None  # Character to indicate comments
    quoting: int = 0  # QUOTE_MINIMAL
    
    # Component configurations
    missing: MissingValueConfig = field(default_factory=MissingValueConfig)
    outliers: OutlierConfig = field(default_factory=OutlierConfig)
    datatypes: DataTypeConfig = field(default_factory=DataTypeConfig)
    duplicates: DuplicateConfig = field(default_factory=DuplicateConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    encoding: EncodingConfig = field(default_factory=EncodingConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    # Column filtering
    columns_to_keep: List[str] = field(default_factory=list)
    columns_to_drop: List[str] = field(default_factory=list)
    
    # Pipeline configuration
    pipeline_steps: List[str] = field(default_factory=lambda: [
        'read_data',
        'detect_types',
        'handle_missing',
        'handle_outliers',
        'handle_duplicates',
        'normalize_data',
        'encode_categories',
        'validate_data',
        'write_data',
        'generate_report'
    ])


#################################################
# Data Profiling
#################################################

class DataProfiler:
    """Generates comprehensive profile of the dataset"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.profile = {}
        self.start_time = time.time()
        
    def profile_dataset(self) -> Dict:
        """Generate complete dataset profile"""
        logger.info("Profiling dataset...")
        
        self.profile = {
            'basic_info': self._get_basic_info(),
            'column_stats': self._get_column_stats(),
            'missing_values': self._analyze_missing_values(),
            'duplicates': self._analyze_duplicates(),
            'outliers': self._analyze_outliers(),
            'correlations': self._analyze_correlations() if len(self.df) > 0 else {},
            'memory_usage': self._get_memory_usage(),
            'profiling_time': time.time() - self.start_time
        }
        
        return self.profile
    
    def _get_basic_info(self) -> Dict:
        """Get basic dataset information"""
        return {
            'rows': len(self.df),
            'columns': len(self.df.columns),
            'num_numeric_columns': len(self.df.select_dtypes(include=np.number).columns),
            'num_categorical_columns': len(self.df.select_dtypes(include=['object', 'category']).columns),
            'num_datetime_columns': len(self.df.select_dtypes(include=['datetime64', 'timedelta64']).columns),
            'num_boolean_columns': len(self.df.select_dtypes(include=['bool']).columns),
            'column_names': self.df.columns.tolist(),
        }
    
    def _get_column_stats(self) -> Dict:
        """Get detailed statistics for each column"""
        stats = {}
        
        for col in self.df.columns:
            col_stats = {'type': str(self.df[col].dtype)}
            
            # Get basic stats for the column
            if pd.api.types.is_numeric_dtype(self.df[col]):
                try:
                    col_stats.update({
                        'min': float(self.df[col].min()) if not pd.isna(self.df[col].min()) else None,
                        'max': float(self.df[col].max()) if not pd.isna(self.df[col].max()) else None,
                        'mean': float(self.df[col].mean()) if not pd.isna(self.df[col].mean()) else None,
                        'median': float(self.df[col].median()) if not pd.isna(self.df[col].median()) else None,
                        'std': float(self.df[col].std()) if not pd.isna(self.df[col].std()) else None,
                        'unique_values': int(self.df[col].nunique()),
                        'zero_count': int((self.df[col] == 0).sum()),
                        'negative_count': int((self.df[col] < 0).sum()),
                    })
                except:
                    pass
            else:
                try:
                    # Get stats for non-numeric columns
                    value_counts = self.df[col].value_counts()
                    
                    col_stats.update({
                        'unique_values': int(self.df[col].nunique()),
                        'most_common': value_counts.index[0] if not value_counts.empty else None,
                        'most_common_count': int(value_counts.iloc[0]) if not value_counts.empty else 0,
                    })
                    
                    # Add top 5 values for categorical
                    if self.df[col].nunique() < 20:
                        col_stats['top_values'] = dict(value_counts.head(5))
                except:
                    pass
            
            # Add missing value info
            col_stats['missing_count'] = int(self.df[col].isna().sum())
            col_stats['missing_percentage'] = float(self.df[col].isna().mean() * 100)
            
            stats[col] = col_stats
        
        return stats
    
    def _analyze_missing_values(self) -> Dict:
        """Analyze missing values in the dataset"""
        missing_cols = self.df.columns[self.df.isna().any()].tolist()
        
        missing_info = {
            'total_missing_values': int(self.df.isna().sum().sum()),
            'total_missing_percentage': float(self.df.isna().stack().mean() * 100) if len(self.df) > 0 else 0,
            'columns_with_missing': missing_cols,
            'rows_with_missing': int((self.df.isna().sum(axis=1) > 0).sum()),
            'rows_with_missing_percentage': float((self.df.isna().sum(axis=1) > 0).mean() * 100) if len(self.df) > 0 else 0,
        }
        
        # Check for possible missing value patterns
        if len(missing_cols) > 0:
            # Create a binary missing pattern DataFrame
            missing_patterns = self.df[missing_cols].isna().astype(int)
            pattern_counts = missing_patterns.value_counts().reset_index()
            
            # Identify top patterns
            top_patterns = []
            for i, pattern in pattern_counts.iterrows():
                if i >= 5:  # Only show top 5 patterns
                    break
                    
                pattern_dict = {}
                for j, col in enumerate(missing_cols):
                    pattern_dict[col] = bool(pattern[col])
                    
                top_patterns.append({
                    'pattern': pattern_dict,
                    'count': int(pattern['count']) if 'count' in pattern else None
                })
                
            missing_info['top_missing_patterns'] = top_patterns
        
        return missing_info
    
    def _analyze_duplicates(self) -> Dict:
        """Analyze duplicated rows"""
        dup_info = {
            'duplicate_rows': int(self.df.duplicated().sum()),
            'duplicate_percentage': float(self.df.duplicated().mean() * 100) if len(self.df) > 0 else 0
        }
        
        # Check for duplicates in individual columns
        col_dups = {}
        for col in self.df.columns:
            col_dups[col] = {
                'duplicate_count': int(self.df.duplicated(subset=[col]).sum()),
                'unique_values': int(self.df[col].nunique()),
                'unique_percentage': float(self.df[col].nunique() / len(self.df) * 100) if len(self.df) > 0 else 0
            }
            
        dup_info['column_duplicates'] = col_dups
        
        return dup_info
    
    def _analyze_outliers(self) -> Dict:
        """Detect outliers in numeric columns"""
        outlier_info = {}
        numeric_cols = self.df.select_dtypes(include=np.number).columns
        
        for col in numeric_cols:
            # Skip columns with too many missing values
            if self.df[col].isna().mean() > 0.5:
                continue
                
            data = self.df[col].dropna()
            
            if len(data) < 10:  # Skip if too few values
                continue
                
            # Basic Z-score outliers
            try:
                z_scores = np.abs(stats.zscore(data))
                outlier_info[col] = {
                    'zscore_outliers_count': int((z_scores > 3).sum()),
                    'zscore_outliers_percentage': float((z_scores > 3).mean() * 100),
                    'min': float(data.min()),
                    'max': float(data.max()),
                    'q1': float(data.quantile(0.25)),
                    'q3': float(data.quantile(0.75)),
                    'iqr': float(data.quantile(0.75) - data.quantile(0.25))
                }
                
                # IQR-based outliers
                q1, q3 = data.quantile(0.25), data.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outlier_info[col]['iqr_outliers_count'] = int(((data < lower_bound) | (data > upper_bound)).sum())
                outlier_info[col]['iqr_outliers_percentage'] = float(((data < lower_bound) | (data > upper_bound)).mean() * 100)
            except:
                continue
                
        return {'numeric_outliers': outlier_info}
    
    def _analyze_correlations(self) -> Dict:
        """Calculate correlations between numeric columns"""
        try:
            numeric_df = self.df.select_dtypes(include=np.number)
            
            if len(numeric_df.columns) < 2:
                return {}
                
            # Calculate correlations
            corr_matrix = numeric_df.corr().round(3)
            
            # Get top correlations
            corrs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i):
                    val = corr_matrix.iloc[i, j]
                    if not pd.isna(val):
                        corrs.append({
                            'column1': corr_matrix.index[i],
                            'column2': corr_matrix.columns[j],
                            'correlation': float(val)
                        })
            
            # Sort by absolute correlation value
            sorted_corrs = sorted(corrs, key=lambda x: abs(x['correlation']), reverse=True)
            
            return {
                'top_correlations': sorted_corrs[:10],
                'correlation_matrix': corr_matrix.to_dict()
            }
        except:
            return {}
    
    def _get_memory_usage(self) -> Dict:
        """Analyze memory usage of the dataframe"""
        memory_usage = self.df.memory_usage(deep=True)
        
        return {
            'total': int(memory_usage.sum()),
            'per_column': {col: int(memory_usage[i]) for i, col in enumerate(self.df.columns)},
            'total_mb': round(memory_usage.sum() / (1024 * 1024), 2)
        }
    
    def generate_profile_report(self, output_path: str = 'data_profile_report.html'):
        """Generate HTML report from profile data"""
        try:
            import jinja2
            
            # Load the template from string
            template_str = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Data Profile Report</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    h1, h2, h3 { color: #333366; }
                    table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                    th, td { padding: 8px; text-align: left; border: 1px solid #ddd; }
                    th { background-color: #f2f2f2; }
                    .section { margin-bottom: 30px; border: 1px solid #ddd; padding: 20px; border-radius: 5px; }
                    .missing-high { background-color: #ffcccc; }
                    .missing-medium { background-color: #ffffcc; }
                    .warning { color: orange; }
                    .error { color: red; }
                    .ok { color: green; }
                </style>
            </head>
            <body>
                <h1>Data Profile Report</h1>
                <div class="section">
                    <h2>Basic Information</h2>
                    <table>
                        <tr><th>Number of Rows</th><td>{{ profile.basic_info.rows }}</td></tr>
                        <tr><th>Number of Columns</th><td>{{ profile.basic_info.columns }}</td></tr>
                        <tr><th>Numeric Columns</th><td>{{ profile.basic_info.num_numeric_columns }}</td></tr>
                        <tr><th>Categorical Columns</th><td>{{ profile.basic_info.num_categorical_columns }}</td></tr>
                        <tr><th>DateTime Columns</th><td>{{ profile.basic_info.num_datetime_columns }}</td></tr>
                        <tr><th>Boolean Columns</th><td>{{ profile.basic_info.num_boolean_columns }}</td></tr>
                        <tr><th>Total Missing Values</th><td>{{ profile.missing_values.total_missing_values }} ({{ profile.missing_values.total_missing_percentage|round(2) }}%)</td></tr>
                        <tr><th>Duplicate Rows</th><td>{{ profile.duplicates.duplicate_rows }} ({{ profile.duplicates.duplicate_percentage|round(2) }}%)</td></tr>
                        <tr><th>Memory Usage</th><td>{{ profile.memory_usage.total_mb }} MB</td></tr>
                    </table>
                </div>
                
                <div class="section">
                    <h2>Column Details</h2>
                    <table>
                        <tr>
                            <th>Column</th>
                            <th>Type</th>
                            <th>Missing</th>
                            <th>Unique</th>
                            <th>Min</th>
                            <th>Max</th>
                            <th>Mean</th>
                            <th>Std</th>
                        </tr>
                        {% for col, stats in profile.column_stats.items() %}
                        <tr>
                            <td>{{ col }}</td>
                            <td>{{ stats.type }}</td>
                            <td class="{% if stats.missing_percentage > 20 %}missing-high{% elif stats.missing_percentage > 5 %}missing-medium{% endif %}">
                                {{ stats.missing_count }} ({{ stats.missing_percentage|round(2) }}%)
                            </td>
                            <td>{% if stats.unique_values %}{{ stats.unique_values }}{% else %}-{% endif %}</td>
                            <td>{% if stats.min is defined %}{{ stats.min }}{% else %}-{% endif %}</td>
                            <td>{% if stats.max is defined %}{{ stats.max }}{% else %}-{% endif %}</td>
                            <td>{% if stats.mean is defined %}{{ stats.mean|round(2) }}{% else %}-{% endif %}</td>
                            <td>{% if stats.std is defined %}{{ stats.std|round(2) }}{% else %}-{% endif %}</td>
                        </tr>
                        {% endfor %}
                    </table>
                </div>
                
                {% if profile.missing_values.columns_with_missing %}
                <div class="section">
                    <h2>Missing Value Analysis</h2>
                    <p>{{ profile.missing_values.rows_with_missing }} rows ({{ profile.missing_values.rows_with_missing_percentage|round(2) }}%) contain at least one missing value.</p>
                    
                    <h3>Columns with Missing Values</h3>
                    <table>
                        <tr>
                            <th>Column</th>
                            <th>Missing Count</th>
                            <th>Missing Percentage</th>
                        </tr>
                        {% for col in profile.missing_values.columns_with_missing %}
                        <tr>
                            <td>{{ col }}</td>
                            <td>{{ profile.column_stats[col].missing_count }}</td>
                            <td>{{ profile.column_stats[col].missing_percentage|round(2) }}%</td>
                        </tr>
                        {% endfor %}
                    </table>
                </div>
                {% endif %}
                
                {% if profile.outliers.numeric_outliers %}
                <div class="section">
                    <h2>Outlier Analysis</h2>
                    <table>
                        <tr>
                            <th>Column</th>
                            <th>Z-score Outliers</th>
                            <th>IQR Outliers</th>
                            <th>Min</th>
                            <th>Q1</th>
                            <th>Q3</th>
                            <th>Max</th>
                        </tr>
                        {% for col, stats in profile.outliers.numeric_outliers.items() %}
                        <tr>
                            <td>{{ col }}</td>
                            <td>{{ stats.zscore_outliers_count }} ({{ stats.zscore_outliers_percentage|round(2) }}%)</td>
                            <td>{{ stats.iqr_outliers_count }} ({{ stats.iqr_outliers_percentage|round(2) }}%)</td>
                            <td>{{ stats.min }}</td>
                            <td>{{ stats.q1 }}</td>
                            <td>{{ stats.q3 }}</td>
                            <td>{{ stats.max }}</td>
                        </tr>
                        {% endfor %}
                    </table>
                </div>
                {% endif %}
                
                {% if profile.correlations and profile.correlations.top_correlations %}
                <div class="section">
                    <h2>Correlation Analysis</h2>
                    <h3>Top Correlations</h3>
                    <table>
                        <tr>
                            <th>Column 1</th>
                            <th>Column 2</th>
                            <th>Correlation</th>
                        </tr>
                        {% for corr in profile.correlations.top_correlations %}
                        <tr>
                            <td>{{ corr.column1 }}</td>
                            <td>{{ corr.column2 }}</td>
                            <td>{{ corr.correlation }}</td>
                        </tr>
                        {% endfor %}
                    </table>
                </div>
                {% endif %}
                
                <div class="section">
                    <h2>Data Quality Issues</h2>
                    <ul>
                        {% if profile.missing_values.total_missing_percentage > 10 %}
                        <li class="error">High percentage of missing values ({{ profile.missing_values.total_missing_percentage|round(2) }}%)</li>
                        {% endif %}
                        
                        {% if profile.duplicates.duplicate_percentage > 5 %}
                        <li class="warning">High percentage of duplicate rows ({{ profile.duplicates.duplicate_percentage|round(2) }}%)</li>
                        {% endif %}
                        
                        {% for col, stats in profile.column_stats.items() %}
                            {% if stats.missing_percentage > 20 %}
                            <li class="warning">Column '{{ col }}' has {{ stats.missing_percentage|round(2) }}% missing values</li>
                            {% endif %}
                            
                            {% if stats.type.startswith('int') or stats.type.startswith('float') %}
                                {% if profile.outliers.numeric_outliers.get(col, {}).get('iqr_outliers_percentage', 0) > 10 %}
                                <li class="warning">Column '{{ col }}' has a high percentage of outliers ({{ profile.outliers.numeric_outliers[col].iqr_outliers_percentage|round(2) }}%)</li>
                                {% endif %}
                            {% endif %}
                        {% endfor %}
                        
                        {% if profile.basic_info.num_numeric_columns == 0 %}
                        <li class="warning">No numeric columns detected</li>
                        {% endif %}
                    </ul>
                </div>
                
                <div class="footer">
                    <p>Generated on {{ now }} • Processing time: {{ profile.profiling_time|round(2) }} seconds</p>
                </div>
            </body>
            </html>
            """
            
            # Create Jinja template
            template = jinja2.Template(template_str)
            
            # Render template with profile data
            html = template.render(profile=self.profile, now=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            
            # Write to file
            with open(output_path, 'w') as f:
                f.write(html)
                
            logger.info(f"Profile report written to {output_path}")
            
        except Exception as e:
            logger.error(f"Error generating profile report: {str(e)}")
            
            # Fallback to JSON report
            with open(output_path.replace('.html', '.json'), 'w') as f:
                json.dump(self.profile, f, indent=2)
            
            logger.info(f"Fallback profile JSON written to {output_path.replace('.html', '.json')}")


#################################################
# Data Cleaning Components
#################################################

class MissingValueHandler:
    """Handles missing values in the dataframe"""
    
    def __init__(self, config: MissingValueConfig):
        self.config = config
        self.imputer = None
        self.stats = {}
        
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process missing values according to configuration"""
        if df.empty:
            return df
            
        logger.info(f"Handling missing values with strategy: {self.config.strategy}")
        
        # First, detect columns and rows with high missing value proportions
        row_missing_ratio = df.isna().mean(axis=1)
        col_missing_ratio = df.isna().mean(axis=0)
        
        # Setup stats
        self.stats = {
            'total_missing_before': int(df.isna().sum().sum()),
            'missing_percentage_before': float(df.isna().stack().mean() * 100) if len(df) > 0 else 0,
            'rows_dropped': 0,
            'columns_dropped': 0,
        }
        
        # Auto-strategy selection
        if self.config.strategy == 'auto':
            # If almost all values are missing in many columns, better to drop columns
            if (col_missing_ratio > 0.7).sum() > len(df.columns) * 0.3:
                strategy = 'drop'
            # If missing values are moderate, imputation is better
            elif df.isna().mean().mean() < 0.3:
                strategy = 'impute'
            # Default strategy is drop
            else:
                strategy = 'drop'
        else:
            strategy = self.config.strategy
        
        # Apply selected strategy
        if strategy == 'drop':
            # Drop columns with high missing value proportions
            cols_to_drop = col_missing_ratio[col_missing_ratio > self.config.threshold_column].index.tolist()
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)
                logger.info(f"Dropped {len(cols_to_drop)} columns with high missing values: {cols_to_drop}")
                self.stats['columns_dropped'] = len(cols_to_drop)
                self.stats['dropped_column_names'] = cols_to_drop
            
            # Drop rows with high missing value proportions
            rows_to_drop = row_missing_ratio[row_missing_ratio > self.config.threshold_row].index.tolist()
            if rows_to_drop:
                df = df.drop(index=rows_to_drop)
                logger.info(f"Dropped {len(rows_to_drop)} rows with high missing values")
                self.stats['rows_dropped'] = len(rows_to_drop)
            
            # For remaining missing values, use imputation
            if df.isna().any().any():
                df = self._impute_missing(df)
                
        elif strategy == 'impute':
            # Direct imputation of all missing values
            df = self._impute_missing(df)
            
        elif strategy == 'flag':
            # Add indicator columns for all columns with missing values
            missing_cols = df.columns[df.isna().any()].tolist()
            for col in missing_cols:
                df[f"{col}_ismissing"] = df[col].isna().astype(int)
                
            # Then impute the missing values
            df = self._impute_missing(df)
            
        # Calculate final stats
        self.stats['total_missing_after'] = int(df.isna().sum().sum())
        self.stats['missing_percentage_after'] = float(df.isna().stack().mean() * 100) if len(df) > 0 else 0
        
        return df
    
    def _impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values using the configured method"""
        # Auto method selection based on data types
        if self.config.imputation_method == 'auto':
            # For mixed data types, use different strategies
            if len(df.select_dtypes(include=np.number).columns) > 0:
                # If there are numeric columns and reasonable amount of data, use iterative imputation
                if len(df) > 100 and df.shape[1] < 50:
                    numeric_method = 'iterative'
                else:
                    numeric_method = 'median'
            else:
                numeric_method = 'median'
                
            # For categorical/text columns
            categorical_method = 'mode'
        else:
            numeric_method = self.config.imputation_method
            categorical_method = self.config.imputation_method
        
        # Split by data type
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
        
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Process numeric columns
        if numeric_cols and any(df[numeric_cols].isna().any()):
            logger.info(f"Imputing numeric missing values using {numeric_method} method")
            
            if numeric_method == 'knn':
                imputer = KNNImputer(n_neighbors=self.config.knn_neighbors)
                result_df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                
            elif numeric_method == 'iterative':
                try:
                    imputer = IterativeImputer(max_iter=10, random_state=42)
                    result_df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                except:
                    # Fallback to median if iterative fails
                    logger.warning("Iterative imputation failed, falling back to median")
                    result_df[numeric_cols] = result_df[numeric_cols].fillna(result_df[numeric_cols].median())
                    
            elif numeric_method == 'mean':
                result_df[numeric_cols] = result_df[numeric_cols].fillna(result_df[numeric_cols].mean())
                
            elif numeric_method == 'median':
                result_df[numeric_cols] = result_df[numeric_cols].fillna(result_df[numeric_cols].median())
                
            elif numeric_method == 'constant':
                result_df[numeric_cols] = result_df[numeric_cols].fillna(self.config.fill_value)
        
        # Process categorical/text columns
        if categorical_cols and any(df[categorical_cols].isna().any()):
            logger.info(f"Imputing categorical missing values using {categorical_method} method")
            
            if categorical_method == 'mode':
                for col in categorical_cols:
                    if df[col].isna().any():
                        mode_value = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                        result_df[col] = result_df[col].fillna(mode_value)
                        
            elif categorical_method == 'constant':
                fill_value = 'Unknown' if isinstance(self.config.fill_value, (int, float)) else self.config.fill_value
                result_df[categorical_cols] = result_df[categorical_cols].fillna(fill_value)
        
        # Add indicator columns if configured
        if self.config.add_indicator:
            missing_cols = df.columns[df.isna().any()].tolist()
            for col in missing_cols:
                result_df[f"{col}_ismissing"] = df[col].isna().astype(int)
        
        return result_df


class OutlierHandler:
    """Detects and handles outliers in numeric columns"""
    
    def __init__(self, config: OutlierConfig):
        self.config = config
        self.models = {}
        self.stats = {}
        
    def _detect_outliers_zscore(self, df: pd.DataFrame, columns: List[str]) -> Dict[str, np.ndarray]:
        """Detect outliers using Z-score method"""
        outliers = {}
        
        for col in columns:
            if df[col].isna().all():
                continue
                
            data = df[col].dropna()
            
            if len(data) < 10:
                continue
                
            try:
                z_scores = np.abs(stats.zscore(data))
                outlier_mask = z_scores > self.config.threshold
                
                # Create a full-sized mask matching the original series
                full_mask = pd.Series(False, index=df.index)
                full_mask[data.index] = outlier_mask
                
                outliers[col] = full_mask.values
                
                self.stats[col] = {
                    'method': 'zscore',
                    'threshold': self.config.threshold,
                    'outlier_count': int(outlier_mask.sum()),
                    'outlier_percentage': float(outlier_mask.mean() * 100)
                }
            except:
                continue
                
        return outliers
    
    def _detect_outliers_iqr(self, df: pd.DataFrame, columns: List[str]) -> Dict[str, np.ndarray]:
        """Detect outliers using IQR method"""
        outliers = {}
        
        for col in columns:
            if df[col].isna().all():
                continue
                
            data = df[col].dropna()
            
            if len(data) < 10:
                continue
                
            try:
                q1 = data.quantile(0.25)
                q3 = data.quantile(0.75)
                iqr = q3 - q1
                
                lower_bound = q1 - self.config.iqr_multiplier * iqr
                upper_bound = q3 + self.config.iqr_multiplier * iqr
                
                outlier_mask = (data < lower_bound) | (data > upper_bound)
                
                # Create a full-sized mask matching the original series
                full_mask = pd.Series(False, index=df.index)
                full_mask[data.index] = outlier_mask
                
                outliers[col] = full_mask.values
                
                self.stats[col] = {
                    'method': 'iqr',
                    'multiplier': self.config.iqr_multiplier,
                    'q1': float(q1),
                    'q3': float(q3),
                    'iqr': float(iqr),
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound),
                    'outlier_count': int(outlier_mask.sum()),
                    'outlier_percentage': float(outlier_mask.mean() * 100)
                }
            except:
                continue
                
        return outliers
    
    def _detect_outliers_isolation_forest(self, df: pd.DataFrame, columns: List[str]) -> Dict[str, np.ndarray]:
        """Detect outliers using Isolation Forest"""
        outliers = {}
        
        # Group columns to use Isolation Forest most effectively
        # Isolation Forest works best with multiple variables
        numeric_df = df[columns].copy()
        
        # Fill missing values temporarily for detection
        numeric_df = numeric_df.fillna(numeric_df.mean())
        
        # Skip if too few rows
        if len(numeric_df) < 20 or numeric_df.empty:
            return outliers
            
        try:
            model = IsolationForest(
                contamination=self.config.contamination,
                random_state=42,
                n_jobs=-1
            )
            
            # Fit the model and predict
            preds = model.fit_predict(numeric_df)
            
            # Convert to boolean mask (outliers are -1, inliers are 1)
            outlier_mask = preds == -1
            
            # Store the model for possible reuse
            self.models['isolation_forest'] = model
            
            # Create individual column results
            for col in columns:
                outliers[col] = outlier_mask
                
                self.stats[col] = {
                    'method': 'isolation_forest',
                    'contamination': self.config.contamination,
                    'outlier_count': int(outlier_mask.sum()),
                    'outlier_percentage': float(outlier_mask.mean() * 100)
                }
        except Exception as e:
            logger.warning(f"Isolation Forest failed: {str(e)}")
            
        return outliers
    
    def handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and handle outliers in numeric columns"""
        if df.empty:
            return df
            
        logger.info(f"Handling outliers with method: {self.config.detection_method}")
        
        # Determine which columns to check
        if self.config.columns_to_check:
            # Filter specified columns to ensure they exist and are numeric
            columns = [col for col in self.config.columns_to_check 
                      if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
        else:
            # Use all numeric columns
            columns = df.select_dtypes(include=np.number).columns.tolist()
        
        if not columns:
            logger.info("No numeric columns to check for outliers")
            return df
        
        # Don't process columns with too many missing values
        columns = [col for col in columns if df[col].isna().mean() < 0.3]
        
        # Auto method selection
        if self.config.detection_method == 'auto':
            # Choose method based on data characteristics
            if len(df) > 1000 and len(columns) >= 3:
                detection_method = 'isolation_forest'
            else:
                detection_method = 'iqr'
        else:
            detection_method = self.config.detection_method
        
        # Detect outliers
        if detection_method == 'zscore':
            outliers = self._detect_outliers_zscore(df, columns)
        elif detection_method == 'iqr':
            outliers = self._detect_outliers_iqr(df, columns)
        elif detection_method == 'isolation_forest':
            outliers = self._detect_outliers_isolation_forest(df, columns)
        elif detection_method == 'none':
            return df
        else:
            logger.warning(f"Unknown outlier detection method: {detection_method}")
            return df
        
        # Skip if no outliers detected
        if not outliers:
            logger.info("No outliers detected")
            return df
        
        # Auto handling strategy selection
        if self.config.handling_strategy == 'auto':
            # For a smaller percentage of outliers, removal might be OK
            if all(self.stats.get(col, {}).get('outlier_percentage', 0) < 5 for col in outliers.keys()):
                handling_strategy = 'remove'
            else:
                handling_strategy = 'cap'
        else:
            handling_strategy = self.config.handling_strategy
        
        # Handle outliers
        result_df = df.copy()
        
        total_outliers = 0
        
        for col, mask in outliers.items():
            num_outliers = mask.sum()
            total_outliers += num_outliers
            
            if num_outliers == 0:
                continue
                
            logger.info(f"Found {num_outliers} outliers in column '{col}' ({self.stats[col]['outlier_percentage']:.2f}%)")
            
            if handling_strategy == 'remove':
                # Only remove if it's not too many
                if self.stats[col]['outlier_percentage'] < 10:
                    result_df = result_df[~mask]
                    logger.info(f"Removed {num_outliers} outlier rows from column '{col}'")
                    
            elif handling_strategy == 'cap':
                # Cap to range limits
                if 'upper_bound' in self.stats[col] and 'lower_bound' in self.stats[col]:
                    # For IQR method, we have the bounds
                    upper = self.stats[col]['upper_bound']
                    lower = self.stats[col]['lower_bound']
                else:
                    # For other methods, calculate percentiles
                    upper = np.percentile(df[col].dropna(), 99.5)
                    lower = np.percentile(df[col].dropna(), 0.5)
                
                # Cap the values
                result_df.loc[mask & (result_df[col] > upper), col] = upper
                result_df.loc[mask & (result_df[col] < lower), col] = lower
                logger.info(f"Capped {num_outliers} outliers in column '{col}' to range [{lower}, {upper}]")
                
            elif handling_strategy == 'flag':
                # Add indicator column
                result_df[f"{col}_is_outlier"] = mask.astype(int)
                logger.info(f"Flagged {num_outliers} outliers in column '{col}'")
        
        logger.info(f"Total outliers handled: {total_outliers}")
        
        return result_df


class DataTypeManager:
    """Manages and converts data types in the dataframe"""
    
    def __init__(self, config: DataTypeConfig):
        self.config = config
        self.conversions = {}
        
    def _infer_datetime_format(self, column: pd.Series) -> str:
        """Infer the datetime format from a column of string values"""
        # Check the first non-null value
        sample = column.dropna().iloc[0] if not column.dropna().empty else ""
        
        # Common date formats to try
        formats = [
            '%Y-%m-%d', '%d-%m-%Y', '%m-%d-%Y',
            '%Y/%m/%d', '%d/%m/%Y', '%m/%d/%Y',
            '%Y-%m-%d %H:%M:%S', '%d-%m-%Y %H:%M:%S', '%m-%d-%Y %H:%M:%S',
            '%Y/%m/%d %H:%M:%S', '%d/%m/%Y %H:%M:%S', '%m/%d/%Y %H:%M:%S'
        ]
        
        for fmt in formats:
            try:
                pd.to_datetime(sample, format=fmt)
                return fmt
            except:
                continue
                
        # If no specific format matches, return 'auto' for pandas to infer
        return 'auto'
    
    def _is_likely_datetime(self, column: pd.Series) -> bool:
        """Check if a column is likely to contain datetime values"""
        if column.dtype.kind in 'M':  # Already datetime
            return True
            
        if column.dtype.kind not in 'OSU':  # Not object or string
            return False
            
        # Sample values for checking
        sample = column.dropna().head(10).astype(str)
        
        # Check if sample values match common date patterns
        date_patterns = [
            r'\d{4}[-/]\d{1,2}[-/]\d{1,2}',  # YYYY-MM-DD
            r'\d{1,2}[-/]\d{1,2}[-/]\d{4}',  # DD-MM-YYYY or MM-DD-YYYY
            r'\d{4}[-/]\d{1,2}[-/]\d{1,2} \d{1,2}:\d{1,2}',  # With time
            r'\d{1,2}[-/]\d{1,2}[-/]\d{4} \d{1,2}:\d{1,2}',  # With time
        ]
        
        # Check if at least 80% of non-null values match any pattern
        pattern_matches = 0
        for value in sample:
            if any(re.match(pattern, value) for pattern in date_patterns):
                pattern_matches += 1
                
        return pattern_matches >= max(1, 0.8 * len(sample))
    
    def _is_likely_boolean(self, column: pd.Series) -> bool:
        """Check if a column is likely to contain boolean values"""
        if column.dtype.kind == 'b':  # Already boolean
            return True
            
        if column.dtype.kind not in 'OSUi':  # Not object, string, or integer
            return False
            
        # Check unique values
        unique_vals = set(column.dropna().astype(str).str.lower())
        
        # Boolean indicators
        true_indicators = set(['yes', 'true', 'y', 't', '1', 'on', 'enable', 'enabled'])
        false_indicators = set(['no', 'false', 'n', 'f', '0', 'off', 'disable', 'disabled'])
        
        # If all unique values are in true/false indicators
        return (unique_vals.issubset(true_indicators | false_indicators) and
                unique_vals & true_indicators and
                unique_vals & false_indicators)
    
    def _is_likely_categorical(self, column: pd.Series) -> bool:
        """Check if a column is likely to be categorical"""
        if column.dtype.name == 'category':  # Already categorical
            return True
            
        if column.dtype.kind not in 'OSU':  # Not object or string
            return False
            
        # Heuristic: If the column has relatively few unique values compared to its length
        unique_count = column.nunique()
        return unique_count <= min(50, 0.1 * len(column))
    
    def detect_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and convert column types based on content"""
        if df.empty:
            return df
            
        if not self.config.auto_convert:
            return df
            
        logger.info("Detecting and converting data types")
        
        result_df = df.copy()
        
        # Apply manual type mappings first
        for col, dtype in self.config.type_mappings.items():
            if col in result_df.columns:
                try:
                    result_df[col] = result_df[col].astype(dtype)
                    self.conversions[col] = {'from': str(df[col].dtype), 'to': dtype, 'method': 'manual'}
                except:
                    logger.warning(f"Failed to convert column '{col}' to type '{dtype}'")
        
        # Automatic conversions
        for col in result_df.columns:
            # Skip manually mapped columns
            if col in self.config.type_mappings:
                continue
                
            original_type = str(result_df[col].dtype)
            
            # Handle datetime columns
            if self._is_likely_datetime(result_df[col]):
                try:
                    if self.config.date_format == 'auto':
                        format = self._infer_datetime_format(result_df[col])
                    else:
                        format = self.config.date_format
                    
                    if format == 'auto':
                        result_df[col] = pd.to_datetime(result_df[col], errors='coerce')
                    else:
                        result_df[col] = pd.to_datetime(result_df[col], format=format, errors='coerce')
                        
                    self.conversions[col] = {
                        'from': original_type, 
                        'to': 'datetime64',
                        'method': 'datetime_conversion',
                        'format': format
                    }
                    
                    logger.info(f"Converted column '{col}' from {original_type} to datetime")
                except:
                    logger.warning(f"Failed to convert column '{col}' to datetime")
            
            # Handle boolean columns
            elif self._is_likely_boolean(result_df[col]):
                try:
                    # Define true and false values
                    true_values = self.config.true_values
                    false_values = self.config.false_values
                    
                    # Convert to boolean
                    result_df[col] = result_df[col].astype(str).str.lower().isin(true_values)
                    
                    self.conversions[col] = {
                        'from': original_type, 
                        'to': 'bool',
                        'method': 'boolean_conversion',
                        'true_values': true_values,
                        'false_values': false_values
                    }
                    
                    logger.info(f"Converted column '{col}' from {original_type} to boolean")
                except:
                    logger.warning(f"Failed to convert column '{col}' to boolean")
            
            # Handle categorical columns
            elif self._is_likely_categorical(result_df[col]):
                try:
                    result_df[col] = result_df[col].astype('category')
                    
                    self.conversions[col] = {
                        'from': original_type, 
                        'to': 'category',
                        'method': 'categorical_conversion',
                        'categories': result_df[col].cat.categories.tolist()
                    }
                    
                    logger.info(f"Converted column '{col}' from {original_type} to category")
                except:
                    logger.warning(f"Failed to convert column '{col}' to category")
            
            # Try numeric conversion
            elif self.config.convert_numeric and result_df[col].dtype.kind in 'OSU':
                try:
                    numeric_col = pd.to_numeric(result_df[col], errors='coerce')
                    
                    # Only convert if not too many values are lost
                    null_count_before = result_df[col].isna().sum()
                    null_count_after = numeric_col.isna().sum()
                    
                    # If the conversion doesn't introduce too many NaNs
                    if (null_count_after - null_count_before) < 0.1 * len(result_df):
                        result_df[col] = numeric_col
                        
                        # Determine if int or float
                        if pd.isna(numeric_col).any() or (numeric_col % 1 != 0).any():
                            dtype = 'float64'
                        else:
                            dtype = 'int64'
                            result_df[col] = result_df[col].astype(dtype)
                        
                        self.conversions[col] = {
                            'from': original_type, 
                            'to': dtype,
                            'method': 'numeric_conversion',
                            'null_before': int(null_count_before),
                            'null_after': int(null_count_after)
                        }
                        
                        logger.info(f"Converted column '{col}' from {original_type} to {dtype}")
                except:
                    logger.warning(f"Failed to convert column '{col}' to numeric")
        
        # Apply infer_objects to clean up any remaining objects
        if self.config.infer_objects:
            result_df = result_df.infer_objects()
        
        # Final memory optimization
        for col in result_df.columns:
            # Downcast numeric columns to save memory
            if pd.api.types.is_integer_dtype(result_df[col]):
                # Determine the smallest integer dtype that can hold the data
                result_df[col] = pd.to_numeric(result_df[col], downcast='integer')
            elif pd.api.types.is_float_dtype(result_df[col]):
                # Downcast floats
                result_df[col] = pd.to_numeric(result_df[col], downcast='float')
        
        return result_df


class DuplicateHandler:
    """Handles duplicate rows in the dataframe"""
    
    def __init__(self, config: DuplicateConfig):
        self.config = config
        self.stats = {}
        
    def handle_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and handle duplicate rows based on configuration"""
        if df.empty:
            return df
            
        # Determine subset of columns to check
        if self.config.subset:
            # Filter to only include columns that exist in the dataframe
            subset = [col for col in self.config.subset if col in df.columns]
            if not subset:
                logger.warning("No valid columns in duplicate subset, using all columns")
                subset = None
        else:
            subset = None
            
        # Check for duplicates
        if subset:
            is_duplicate = df.duplicated(subset=subset, keep=False)
        else:
            is_duplicate = df.duplicated(keep=False)
            
        duplicate_count = is_duplicate.sum()
        
        self.stats = {
            'duplicate_count': int(duplicate_count),
            'duplicate_percentage': float(duplicate_count / len(df) * 100) if len(df) > 0 else 0,
            'duplicate_rows_removed': 0
        }
        
        logger.info(f"Found {duplicate_count} duplicate rows ({self.stats['duplicate_percentage']:.2f}%)")
        
        # Auto strategy selection
        if self.config.strategy == 'auto':
            if self.stats['duplicate_percentage'] < 1:
                strategy = 'remove_first'
            elif self.stats['duplicate_percentage'] < 10:
                strategy = 'remove_first'
            else:
                strategy = 'flag'
        else:
            strategy = self.config.strategy
            
        # If no duplicates or strategy is 'none', return original df
        if duplicate_count == 0 or strategy == 'none':
            return df
            
        # Apply strategy
        result_df = df.copy()
        
        if strategy in ['remove_first', 'remove_last', 'remove_all']:
            if strategy == 'remove_first':
                keep = 'last'
            elif strategy == 'remove_last':
                keep = 'first'
            else:  # remove_all
                keep = False
                
            # Remove duplicates
            result_df = result_df.drop_duplicates(subset=subset, keep=keep)
            
            rows_removed = len(df) - len(result_df)
            self.stats['duplicate_rows_removed'] = int(rows_removed)
            
            logger.info(f"Removed {rows_removed} duplicate rows")
            
        elif strategy == 'flag':
            # Add indicator column
            result_df['is_duplicate'] = is_duplicate.astype(int)
            logger.info(f"Flagged {duplicate_count} duplicate rows")
        
        return result_df


class DataNormalizer:
    """Normalizes/standardizes numeric columns"""
    
    def __init__(self, config: NormalizationConfig):
        self.config = config
        self.scalers = {}
        self.stats = {}
        
    def normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize numeric columns according to configuration"""
        if df.empty:
            return df
            
        # Skip if normalization method is 'none'
        if self.config.method == 'none':
            return df
            
        logger.info(f"Normalizing data with method: {self.config.method}")
        
        # Determine which columns to normalize
        if self.config.columns:
            # Filter to ensure columns exist and are numeric
            columns = [col for col in self.config.columns 
                      if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
        else:
            # Use all numeric columns
            columns = df.select_dtypes(include=np.number).columns.tolist()
            
        if not columns:
            logger.info("No numeric columns to normalize")
            return df
            
        # Skip columns with too many missing values
        columns = [col for col in columns if df[col].isna().mean() < 0.3]
        
        # Create a copy of the dataframe
        result_df = df.copy()
        
        # Apply the selected normalization method
        if self.config.method == 'standardize':
            for col in columns:
                scaler = StandardScaler()
                # Handle missing values for fitting
                col_data = df[col].dropna().values.reshape(-1, 1)
                
                if len(col_data) == 0:
                    continue
                    
                scaler.fit(col_data)
                
                # Store stats
                self.stats[col] = {
                    'method': 'standardize',
                    'mean': float(scaler.mean_[0]),
                    'std': float(scaler.scale_[0]),
                    'original_min': float(df[col].min()),
                    'original_max': float(df[col].max())
                }
                
                # Transform only non-missing values
                mask = ~df[col].isna()
                result_df.loc[mask, col] = scaler.transform(df.loc[mask, col].values.reshape(-1, 1)).flatten()
                
                # Save the scaler
                self.scalers[col] = scaler
                
        elif self.config.method == 'minmax':
            for col in columns:
                # Check if custom range is specified
                if col in self.config.custom_ranges:
                    feature_range = self.config.custom_ranges[col]
                else:
                    feature_range = (0, 1)
                    
                scaler = MinMaxScaler(feature_range=feature_range)
                
                # Handle missing values for fitting
                col_data = df[col].dropna().values.reshape(-1, 1)
                
                if len(col_data) == 0:
                    continue
                    
                scaler.fit(col_data)
                
                # Store stats
                self.stats[col] = {
                    'method': 'minmax',
                    'feature_range': feature_range,
                    'data_min': float(scaler.data_min_[0]),
                    'data_max': float(scaler.data_max_[0]),
                    'original_min': float(df[col].min()),
                    'original_max': float(df[col].max())
                }
                
                # Transform only non-missing values
                mask = ~df[col].isna()
                result_df.loc[mask, col] = scaler.transform(df.loc[mask, col].values.reshape(-1, 1)).flatten()
                
                # Save the scaler
                self.scalers[col] = scaler
                
        elif self.config.method == 'robust':
            for col in columns:
                scaler = RobustScaler()
                
                # Handle missing values for fitting
                col_data = df[col].dropna().values.reshape(-1, 1)
                
                if len(col_data) == 0:
                    continue
                    
                scaler.fit(col_data)
                
                # Store stats
                q1 = float(np.percentile(col_data, 25))
                q3 = float(np.percentile(col_data, 75))
                
                self.stats[col] = {
                    'method': 'robust',
                    'center': float(scaler.center_[0]),
                    'scale': float(scaler.scale_[0]),
                    'q1': q1,
                    'q3': q3,
                    'iqr': q3 - q1,
                    'original_min': float(df[col].min()),
                    'original_max': float(df[col].max())
                }
                
                # Transform only non-missing values
                mask = ~df[col].isna()
                result_df.loc[mask, col] = scaler.transform(df.loc[mask, col].values.reshape(-1, 1)).flatten()
                
                # Save the scaler
                self.scalers[col] = scaler
                
        elif self.config.method == 'log':
            for col in columns:
                # Ensure all values are positive
                min_val = df[col].min()
                
                if min_val <= 0:
                    # Add offset to make all values positive
                    offset = abs(min_val) + 1
                else:
                    offset = 0
                    
                # Apply log transformation to non-missing values
                mask = ~df[col].isna()
                result_df.loc[mask, col] = np.log(df.loc[mask, col] + offset)
                
                # Store stats
                self.stats[col] = {
                    'method': 'log',
                    'offset': float(offset),
                    'original_min': float(df[col].min()),
                    'original_max': float(df[col].max()),
                    'transformed_min': float(result_df[col].min()),
                    'transformed_max': float(result_df[col].max())
                }
        
        return result_df


class CategoryEncoder:
    """Encodes categorical variables using various methods"""
    
    def __init__(self, config: EncodingConfig):
        self.config = config
        self.encoders = {}
        self.stats = {}
        
    def encode_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical columns according to configuration"""
        if df.empty:
            return df
            
        # Skip if encoding method is 'none'
        if self.config.method == 'none':
            return df
            
        logger.info(f"Encoding categorical variables with method: {self.config.method}")
        
        # Determine which columns to encode
        if self.config.columns:
            # Filter to ensure columns exist
            columns = [col for col in self.config.columns if col in df.columns]
        else:
            # Use all categorical/object columns
            columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
        if not columns:
            logger.info("No categorical columns to encode")
            return df
            
        # Determine encoding method
        if self.config.method == 'auto':
            # Choose method based on column cardinality
            method_map = {}
            
            for col in columns:
                unique_count = df[col].nunique()
                
                if unique_count <= 2:
                    method_map[col] = 'label'  # Binary variables
                elif unique_count <= self.config.max_categories:
                    method_map[col] = 'one_hot'  # Categorical with few categories
                else:
                    method_map[col] = 'label'  # High cardinality
        else:
            # Use the same method for all columns
            method_map = {col: self.config.method for col in columns}
            
        # Create a copy of the dataframe
        result_df = df.copy()
        
        # Apply encodings
        for col, method in method_map.items():
            if method == 'one_hot':
                self._one_hot_encode(result_df, col)
            elif method == 'label':
                self._label_encode(result_df, col)
            elif method == 'binary':
                self._binary_encode(result_df, col)
            elif method == 'frequency':
                self._frequency_encode(result_df, col)
            elif method == 'target' and self.config.target_column:
                self._target_encode(result_df, col, self.config.target_column)
                
        return result_df
    
    def _one_hot_encode(self, df: pd.DataFrame, col: str) -> None:
        """One-hot encode a column in place"""
        # Get categories
        if df[col].dtype.name == 'category':
            categories = df[col].cat.categories.tolist()
        else:
            categories = df[col].dropna().unique().tolist()
            
        # Create dummy variables
        dummies = pd.get_dummies(df[col], prefix=col, prefix_sep='_', 
                                 dummy_na=False, drop_first=self.config.drop_first)
        
        # Drop original column and add dummies
        df.drop(columns=[col], inplace=True)
        df[dummies.columns] = dummies
        
        # Store encoding info
        self.stats[col] = {
            'method': 'one_hot',
            'categories': categories,
            'drop_first': self.config.drop_first,
            'created_columns': dummies.columns.tolist()
        }
        
        logger.info(f"One-hot encoded column '{col}' creating {len(dummies.columns)} new columns")
    
    def _label_encode(self, df: pd.DataFrame, col: str) -> None:
        """Label encode a column in place"""
        # Handle missing values
        df[f"{col}_missing"] = df[col].isna().astype(int)
        
        # Get unique values
        unique_values = df[col].dropna().unique().tolist()
        
        # Create mapping
        value_map = {val: i for i, val in enumerate(unique_values)}
        
        # Create a new column with encoded values
        new_col = f"{col}_encoded"
        df[new_col] = df[col].map(value_map)
        
        # Fill missing values with -1
        df[new_col] = df[new_col].fillna(-1).astype(int)
        
        # Drop original column
        df.drop(columns=[col], inplace=True)
        
        # Store encoding info
        self.stats[col] = {
            'method': 'label',
            'mapping': value_map,
            'new_column': new_col
        }
        
        logger.info(f"Label encoded column '{col}' to '{new_col}'")
    
    def _binary_encode(self, df: pd.DataFrame, col: str) -> None:
        """Binary encode a column in place"""
        # Get unique values
        unique_values = df[col].dropna().unique().tolist()
        
        # Skip if too few unique values
        if len(unique_values) <= 2:
            self._label_encode(df, col)
            return
            
        # Create mapping
        value_map = {val: i for i, val in enumerate(unique_values)}
        
        # Binary encode
        def to_binary(val):
            if pd.isna(val):
                return None
                
            # Get the integer encoding
            int_val = value_map[val]
            
            # Convert to binary string
            bin_str = bin(int_val)[2:]
            
            # Pad with zeros to fixed width
            width = int(np.log2(len(unique_values))) + 1
            return bin_str.zfill(width)
            
        # Apply encoding
        binary_vals = df[col].apply(to_binary)
        
        # Create binary columns
        width = int(np.log2(len(unique_values))) + 1
        
        binary_cols = []
        for i in range(width):
            new_col = f"{col}_bin_{i}"
            df[new_col] = binary_vals.str[i].fillna(-1).astype(int)
            binary_cols.append(new_col)
            
        # Drop original column
        df.drop(columns=[col], inplace=True)
        
        # Store encoding info
        self.stats[col] = {
            'method': 'binary',
            'mapping': value_map,
            'bit_width': width,
            'created_columns': binary_cols
        }
        
        logger.info(f"Binary encoded column '{col}' creating {len(binary_cols)} new columns")
    
    def _frequency_encode(self, df: pd.DataFrame, col: str) -> None:
        """Frequency encode a column in place"""
        # Calculate frequencies
        value_counts = df[col].value_counts(dropna=False)
        freq_map = value_counts / len(df)
        
        # Create a new column with frequency values
        new_col = f"{col}_freq"
        df[new_col] = df[col].map(freq_map)
        
        # Drop original column
        df.drop(columns=[col], inplace=True)
        
        # Store encoding info
        self.stats[col] = {
            'method': 'frequency',
            'mapping': freq_map.to_dict(),
            'new_column': new_col
        }
        
        logger.info(f"Frequency encoded column '{col}' to '{new_col}'")
    
    def _target_encode(self, df: pd.DataFrame, col: str, target_col: str) -> None:
        """Target encode a column in place"""
        # Check if target column exists
        if target_col not in df.columns:
            logger.warning(f"Target column '{target_col}' not found for target encoding")
            self._label_encode(df, col)
            return
            
        # Check if target is numeric
        if not pd.api.types.is_numeric_dtype(df[target_col]):
            logger.warning(f"Target column '{target_col}' is not numeric, using label encoding instead")
            self._label_encode(df, col)
            return
            
        # Calculate mean target value for each category
        target_means = df.groupby(col)[target_col].mean()
        
        # Create a new column with target means
        new_col = f"{col}_target"
        df[new_col] = df[col].map(target_means)
        
        # Handle missing values with global mean
        global_mean = df[target_col].mean()
        df[new_col] = df[new_col].fillna(global_mean)
        
        # Drop original column
        df.drop(columns=[col], inplace=True)
        
        # Store encoding info
        self.stats[col] = {
            'method': 'target',
            'mapping': target_means.to_dict(),
            'target_column': target_col,
            'global_mean': float(global_mean),
            'new_column': new_col
        }
        
        logger.info(f"Target encoded column '{col}' to '{new_col}' using target '{target_col}'")


class DataValidator:
    """Validates data against defined rules"""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.results = {}
        
    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate the dataframe against defined rules"""
        if df.empty or not self.config.enabled:
            return df
            
        logger.info("Validating data against rules")
        
        result_df = df.copy()
        all_violations = pd.Series(False, index=df.index)
        
        for rule_name, rule_config in self.config.rules.items():
            rule_type = rule_config.get('type')
            columns = rule_config.get('columns', [])
            
            # Filter to only include columns that exist
            columns = [col for col in columns if col in df.columns]
            
            if not columns:
                continue
                
            logger.info(f"Applying rule '{rule_name}' of type '{rule_type}' to {len(columns)} columns")
            
            # Apply the appropriate rule
            if rule_type == 'range':
                min_val = rule_config.get('min')
                max_val = rule_config.get('max')
                violations = self._check_range_rule(df, columns, min_val, max_val)
                
            elif rule_type == 'unique':
                violations = self._check_unique_rule(df, columns)
                
            elif rule_type == 'regex':
                pattern = rule_config.get('pattern')
                violations = self._check_regex_rule(df, columns, pattern)
                
            elif rule_type == 'required':
                violations = self._check_required_rule(df, columns)
                
            elif rule_type == 'type':
                expected_type = rule_config.get('value')
                violations = self._check_type_rule(df, columns, expected_type)
                
            elif rule_type == 'custom':
                expression = rule_config.get('expression')
                violations = self._check_custom_rule(df, expression)
                
            else:
                logger.warning(f"Unknown rule type: {rule_type}")
                continue
                
            # Store results
            violation_count = violations.sum()
            self.results[rule_name] = {
                'rule_type': rule_type,
                'columns': columns,
                'violation_count': int(violation_count),
                'violation_percentage': float(violation_count / len(df) * 100) if len(df) > 0 else 0
            }
            
            logger.info(f"Rule '{rule_name}' found {violation_count} violations " +
                       f"({self.results[rule_name]['violation_percentage']:.2f}%)")
            
            # Update all violations
            all_violations = all_violations | violations
            
            # Handle violations according to configuration
            if violation_count > 0:
                action = rule_config.get('action', self.config.action_on_fail)
                
                if action == 'drop':
                    result_df = result_df[~violations]
                    logger.info(f"Dropped {violation_count} rows that violated rule '{rule_name}'")
                    
                elif action == 'flag':
                    flag_col = f"violates_{rule_name}"
                    result_df[flag_col] = violations.astype(int)
                    logger.info(f"Flagged {violation_count} violations in column '{flag_col}'")
                    
                elif action == 'fix':
                    fix_strategy = rule_config.get('fix_strategy')
                    fix_value = rule_config.get('fix_value')
                    
                    if fix_strategy == 'set_value':
                        # Set a fixed value for violations
                        for col in columns:
                            result_df.loc[violations, col] = fix_value
                            
                    elif fix_strategy == 'to_null':
                        # Set violations to null
                        for col in columns:
                            result_df.loc[violations, col] = None
                            
                    logger.info(f"Fixed {violation_count} violations using strategy '{fix_strategy}'")
                    
                elif action == 'error':
                    if violation_count > 0:
                        error_msg = f"Validation failed: Rule '{rule_name}' has {violation_count} violations"
                        logger.error(error_msg)
                        # Don't actually raise an error, just log it
                        
        # Add overall validation flag
        total_violations = all_violations.sum()
        result_df['validation_passed'] = (~all_violations).astype(int)
        
        logger.info(f"Total violations: {total_violations} ({total_violations / len(df) * 100:.2f}%)")
        
        return result_df
    
    def _check_range_rule(self, df: pd.DataFrame, columns: List[str], min_val, max_val) -> pd.Series:
        """Check if values are within the specified range"""
        violations = pd.Series(False, index=df.index)
        
        for col in columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
                
            col_violations = pd.Series(False, index=df.index)
            
            if min_val is not None:
                col_violations = col_violations | (df[col] < min_val)
                
            if max_val is not None:
                col_violations = col_violations | (df[col] > max_val)
                
            violations = violations | col_violations
            
        return violations
    
    def _check_unique_rule(self, df: pd.DataFrame, columns: List[str]) -> pd.Series:
        """Check if values in columns are unique"""
        return df.duplicated(subset=columns, keep=False)
    
    def _check_regex_rule(self, df: pd.DataFrame, columns: List[str], pattern: str) -> pd.Series:
        """Check if values match the regex pattern"""
        violations = pd.Series(False, index=df.index)
        
        for col in columns:
            if not pd.api.types.is_string_dtype(df[col]):
                continue
                
            # Check if values don't match the pattern
            col_violations = ~df[col].astype(str).str.match(pattern, na=False)
            violations = violations | col_violations
            
        return violations
    
    def _check_required_rule(self, df: pd.DataFrame, columns: List[str]) -> pd.Series:
        """Check if required columns have non-null values"""
        violations = pd.Series(False, index=df.index)
        
        for col in columns:
            col_violations = df[col].isna()
            violations = violations | col_violations
            
        return violations
    
    def _check_type_rule(self, df: pd.DataFrame, columns: List[str], expected_type: str) -> pd.Series:
        """Check if columns have the expected type"""
        violations = pd.Series(False, index=df.index)
        
        for col in columns:
            actual_type = df[col].dtype.name
            
            # Check string vs numeric vs datetime
            if expected_type == 'string' and not pd.api.types.is_string_dtype(df[col]):
                violations = pd.Series(True, index=df.index)
            elif expected_type == 'numeric' and not pd.api.types.is_numeric_dtype(df[col]):
                violations = pd.Series(True, index=df.index)
            elif expected_type == 'datetime' and not pd.api.types.is_datetime64_dtype(df[col]):
                violations = pd.Series(True, index=df.index)
            
        return violations
    
    def _check_custom_rule(self, df: pd.DataFrame, expression: str) -> pd.Series:
        """Evaluate a custom rule using a Python expression"""
        try:
            # Safe evaluation using pandas eval
            return ~pd.eval(expression, target=df)
        except:
            logger.error(f"Failed to evaluate custom rule: {expression}")
            return pd.Series(False, index=df.index)


#################################################
# Data Cleaner
#################################################

class DataCleaner:
    """Main class for data cleaning pipeline"""
    
    def __init__(self, config: CleanerConfig = None):
        self.config = config or CleanerConfig()
        self._initialize_components()
        self.stats = {}
        self.df = None
        self.original_df = None
        
    def _initialize_components(self):
        """Initialize all cleaning components"""
        self.missing_handler = MissingValueHandler(self.config.missing)
        self.outlier_handler = OutlierHandler(self.config.outliers)
        self.type_manager = DataTypeManager(self.config.datatypes)
        self.duplicate_handler = DuplicateHandler(self.config.duplicates)
        self.normalizer = DataNormalizer(self.config.normalization)
        self.encoder = CategoryEncoder(self.config.encoding)
        self.validator = DataValidator(self.config.validation)
        self.profiler = None  # Will be created after data is loaded
        
    def load_config(self, config_path: str) -> None:
        """Load configuration from a YAML or JSON file"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        file_ext = os.path.splitext(config_path)[1].lower()
        
        if file_ext == '.yaml' or file_ext == '.yml':
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        elif file_ext == '.json':
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {file_ext}")
            
        # Create a new config with the loaded values
        self.config = CleanerConfig()
        
        # Update the config recursively
        self._update_config_recursive(self.config, config_dict)
        
        # Re-initialize components with new config
        self._initialize_components()
        
        logger.info(f"Loaded configuration from {config_path}")
        
    def _update_config_recursive(self, config_obj, config_dict):
        """Recursively update config object with values from dict"""
        for key, value in config_dict.items():
            if hasattr(config_obj, key):
                current_value = getattr(config_obj, key)
                
                if isinstance(current_value, (MissingValueConfig, OutlierConfig, DataTypeConfig,
                                             DuplicateConfig, NormalizationConfig, EncodingConfig,
                                             ValidationConfig, PerformanceConfig, OutputConfig)):
                    # Recursively update nested config
                    self._update_config_recursive(current_value, value)
                else:
                    # Directly set the value
                    setattr(config_obj, key, value)
    
    def save_config(self, config_path: str) -> None:
        """Save the current configuration to a file"""
        file_ext = os.path.splitext(config_path)[1].lower()
        
        # Convert config to dict
        config_dict = self._config_to_dict(self.config)
        
        if file_ext == '.yaml' or file_ext == '.yml':
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        elif file_ext == '.json':
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported config file format: {file_ext}")
            
        logger.info(f"Saved configuration to {config_path}")
        
    def _config_to_dict(self, config_obj):
        """Convert config object to dictionary recursively"""
        result = {}
        
        for key, value in config_obj.__dict__.items():
            if isinstance(value, (MissingValueConfig, OutlierConfig, DataTypeConfig,
                                 DuplicateConfig, NormalizationConfig, EncodingConfig,
                                 ValidationConfig, PerformanceConfig, OutputConfig)):
                # Recursively convert nested config
                result[key] = self._config_to_dict(value)
            else:
                # Handle special types
                if isinstance(value, np.ndarray):
                    result[key] = value.tolist()
                elif isinstance(value, pd.Series):
                    result[key] = value.to_dict()
                elif isinstance(value, pd.DataFrame):
                    result[key] = value.to_dict()
                else:
                    result[key] = value
                    
        return result
    
    def _detect_encoding(self, file_path: str) -> str:
        """Detect the file encoding"""
        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)  # Read first 10KB
            
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        
        logger.info(f"Detected encoding: {encoding} (confidence: {result['confidence']:.2f})")
        
        return encoding
    
    def _detect_delimiter(self, file_path: str, encoding: str) -> str:
        """Detect the CSV delimiter"""
        with open(file_path, 'r', encoding=encoding, errors='replace') as f:
            sample = f.read(10000)  # Read first 10KB
            
        # Count potential delimiters
        delimiters = [',', ';', '\t', '|', ':']
        counts = {d: sample.count(d) for d in delimiters}
        
        # Get delimiter with max count
        max_delimiter = max(counts, key=counts.get)
        
        # Only accept if it appears enough times
        if counts[max_delimiter] > 10:
            logger.info(f"Detected delimiter: '{max_delimiter}' (count: {counts[max_delimiter]})")
            return max_delimiter
        else:
            logger.info(f"Could not detect delimiter, defaulting to comma")
            return ','
    
    def read_data(self, file_path: str = None, **kwargs) -> pd.DataFrame:
        """Read data from a CSV file"""
        # Use provided file path or one from config
        file_path = file_path or self.config.input_file
        
        if not file_path:
            raise ValueError("No input file specified")
            
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Input file not found: {file_path}")
            
        logger.info(f"Reading data from {file_path}")
        
        # Auto-detect encoding and delimiter if needed
        encoding = self.config.encoding
        if isinstance(encoding, str) and encoding == 'auto':
            encoding = self._detect_encoding(file_path)
        elif not isinstance(encoding, str):
            encoding = 'utf-8'  # Default to utf-8 if not a string
            
        delimiter = self.config.delimiter
        if delimiter == 'auto':
            delimiter = self._detect_delimiter(file_path, encoding)
        
        # Prepare read parameters
        read_params = {
            'filepath_or_buffer': file_path,
            'delimiter': delimiter,
            'encoding': encoding,
            'header': self.config.header,
            'skiprows': self.config.skip_rows,
            'skipfooter': self.config.skip_footer,
            'na_values': self.config.na_values,
            'true_values': self.config.true_values,
            'false_values': self.config.false_values,
            'comment': self.config.comment,
            'low_memory': self.config.performance.low_memory,
            'quoting': self.config.quoting,
            'engine': 'python' if self.config.skip_footer > 0 else 'c'
        }
        
        # Update with any additional parameters
        read_params.update(kwargs)
        
        # Handle large files with chunking if needed
        if self.config.performance.chunk_size and os.path.getsize(file_path) > 100 * 1024 * 1024:  # 100MB
            logger.info(f"File is large, reading in chunks of {self.config.performance.chunk_size} rows")
            chunks = []
            
            for chunk in pd.read_csv(**read_params, chunksize=self.config.performance.chunk_size):
                chunks.append(chunk)
                
            df = pd.concat(chunks, ignore_index=True)
        else:
            # Read in one go
            df = pd.read_csv(**read_params)
            
        # Filter columns if specified
        if self.config.columns_to_keep:
            # Keep only columns that exist in the dataframe
            cols_to_keep = [col for col in self.config.columns_to_keep if col in df.columns]
            if cols_to_keep:
                df = df[cols_to_keep]
                logger.info(f"Keeping only {len(cols_to_keep)} specified columns")
                
        if self.config.columns_to_drop:
            # Drop only columns that exist in the dataframe
            cols_to_drop = [col for col in self.config.columns_to_drop if col in df.columns]
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)
                logger.info(f"Dropped {len(cols_to_drop)} specified columns")
                
        # Store original dataframe for comparison
        self.original_df = df.copy()
        self.df = df
        
        # Create a profiler
        self.profiler = DataProfiler(df)
        
        # Log basic info
        logger.info(f"Read {len(df)} rows and {len(df.columns)} columns from {file_path}")
        
        return df
    
    def clean_data(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """Execute the full data cleaning pipeline"""
        if df is None:
            if self.df is None:
                raise ValueError("No dataframe to clean. Call read_data() first or provide a dataframe.")
            df = self.df
            
        logger.info("Starting data cleaning pipeline")
        start_time = time.time()
        
        # Record initial stats
        original_rows = len(df)
        original_cols = len(df.columns)
        original_memory = df.memory_usage(deep=True).sum()
        
        pipeline_steps = self.config.pipeline_steps
        
        # Skip steps already done
        if 'read_data' in pipeline_steps and self.original_df is not None:
            pipeline_steps = pipeline_steps[pipeline_steps.index('read_data') + 1:]
            
        # For each step in the pipeline
        for step in pipeline_steps:
            step_start = time.time()
            
            if step == 'read_data':
                # Already handled separately
                continue
                
            elif step == 'detect_types':
                logger.info("Step: Detecting and converting data types")
                df = self.type_manager.detect_types(df)
                
            elif step == 'handle_missing':
                logger.info("Step: Handling missing values")
                df = self.missing_handler.handle_missing_values(df)
                
            elif step == 'handle_outliers':
                logger.info("Step: Handling outliers")
                df = self.outlier_handler.handle_outliers(df)
                
            elif step == 'handle_duplicates':
                logger.info("Step: Handling duplicate rows")
                df = self.duplicate_handler.handle_duplicates(df)
                
            elif step == 'normalize_data':
                logger.info("Step: Normalizing numeric data")
                df = self.normalizer.normalize_data(df)
                
            elif step == 'encode_categories':
                logger.info("Step: Encoding categorical variables")
                df = self.encoder.encode_categories(df)
                
            elif step == 'validate_data':
                logger.info("Step: Validating data")
                df = self.validator.validate_data(df)
                
            elif step == 'write_data':
                # Will be handled separately
                continue
                
            elif step == 'generate_report':
                # Will be handled separately
                continue
                
            else:
                logger.warning(f"Unknown pipeline step: {step}")
                
            step_time = time.time() - step_start
            logger.info(f"Step '{step}' completed in {step_time:.2f} seconds")
            
            # Save intermediate result if configured
            if self.config.output.save_intermediate:
                interim_file = f"interim_{step}_{os.path.basename(self.config.input_file)}"
                self._save_dataframe(df, interim_file)
        
        # Store the cleaned dataframe
        self.df = df
        
        # Record final stats
        final_rows = len(df)
        final_cols = len(df.columns)
        final_memory = df.memory_usage(deep=True).sum()
        
        total_time = time.time() - start_time
        
        self.stats = {
            'original_rows': original_rows,
            'original_columns': original_cols,
            'final_rows': final_rows,
            'final_columns': final_cols,
            'rows_removed': original_rows - final_rows,
            'columns_removed': original_cols - final_cols,
            'memory_before': original_memory,
            'memory_after': final_memory,
            'memory_reduction_pct': (1 - final_memory / original_memory) * 100 if original_memory > 0 else 0,
            'total_processing_time': total_time,
            
            # Include stats from components
            'missing_value_stats': self.missing_handler.stats,
            'outlier_stats': self.outlier_handler.stats,
            'duplicate_stats': self.duplicate_handler.stats,
            'type_conversion_stats': self.type_manager.conversions,
            'normalization_stats': self.normalizer.stats,
            'encoding_stats': self.encoder.stats,
            'validation_stats': self.validator.results
        }
        
        logger.info(f"Data cleaning completed in {total_time:.2f} seconds")
        logger.info(f"Rows: {original_rows} -> {final_rows} ({self.stats['rows_removed']} removed)")
        logger.info(f"Columns: {original_cols} -> {final_cols} ({self.stats['columns_removed']} removed/transformed)")
        
        return df
    
    def write_data(self, output_path: str = None, df: pd.DataFrame = None) -> None:
        """Write the cleaned data to a file"""
        # Use provided df or the one stored in the instance
        if df is None:
            if self.df is None:
                raise ValueError("No dataframe to write. Call clean_data() first or provide a dataframe.")
            df = self.df
            
        # Use provided output path or the one from config
        output_path = output_path or self.config.output_file
        
        if not output_path:
            # Generate a default output path
            if self.config.input_file:
                input_name = os.path.splitext(os.path.basename(self.config.input_file))[0]
                output_path = f"{input_name}{self.config.output.cleaned_suffix}.{self.config.output.output_format}"
            else:
                output_path = f"cleaned_data.{self.config.output.output_format}"
                
        logger.info(f"Writing cleaned data to {output_path}")
        
        self._save_dataframe(df, output_path)
        
        # Save the stats and profile reports if configured
        if self.config.output.create_report:
            self.generate_reports(os.path.dirname(output_path) or '.')
    
    def _save_dataframe(self, df: pd.DataFrame, output_path: str) -> None:
        """Save a dataframe in the configured format"""
        format_ext = os.path.splitext(output_path)[1].lower()
        
        # If format not specified in filename, use the configured format
        if not format_ext or format_ext == '.':
            format_ext = f".{self.config.output.output_format}"
            output_path = f"{output_path}{format_ext}"
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Save in the appropriate format
        if format_ext == '.csv':
            df.to_csv(output_path, index=self.config.output.index)
            
        elif format_ext == '.parquet':
            df.to_parquet(output_path, index=self.config.output.index)
            
        elif format_ext == '.json':
            df.to_json(output_path, orient='records', lines=True)
            
        elif format_ext in ['.xlsx', '.xls']:
            df.to_excel(output_path, index=self.config.output.index)
            
        elif format_ext == '.h5' or format_ext == '.hdf5':
            df.to_hdf(output_path, key='data', mode='w')
            
        elif format_ext == '.pkl' or format_ext == '.pickle':
            df.to_pickle(output_path)
            
        else:
            logger.warning(f"Unknown output format: {format_ext}, defaulting to CSV")
            df.to_csv(output_path, index=self.config.output.index)
            
        # Compress if configured
        if self.config.output.compress_output:
            import gzip
            import shutil
            
            # Compress file
            with open(output_path, 'rb') as f_in:
                with gzip.open(f"{output_path}.gz", 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
                    
            # Remove original
            os.remove(output_path)
            
            logger.info(f"Compressed output to {output_path}.gz")
    
    def generate_reports(self, output_dir: str = '.') -> None:
        """Generate reports on the data cleaning process"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a timestamp for unique filenames
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Base filename from input file or default
        if self.config.input_file:
            base_name = os.path.splitext(os.path.basename(self.config.input_file))[0]
        else:
            base_name = 'data_cleaning'
            
        # 1. Save cleaning stats
        stats_file = os.path.join(output_dir, f"{base_name}_stats_{timestamp}.json")
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2, default=lambda x: str(x) if isinstance(x, (np.integer, np.floating)) else x)
            
        logger.info(f"Saved cleaning stats to {stats_file}")
        
        # 2. Generate data profile reports for before and after
        if self.profiler is not None:
            # Profile the original data
            original_profile = self.profiler.profile_dataset()
            
            # Save original profile report
            if self.config.output.report_format == 'html':
                original_profile_file = os.path.join(output_dir, f"{base_name}_original_profile_{timestamp}.html")
                self.profiler.generate_profile_report(original_profile_file)
            else:
                original_profile_file = os.path.join(output_dir, f"{base_name}_original_profile_{timestamp}.json")
                with open(original_profile_file, 'w') as f:
                    json.dump(original_profile, f, indent=2)
                    
            logger.info(f"Saved original data profile to {original_profile_file}")
            
        # Profile the cleaned data if available
        if self.df is not None:
            cleaned_profiler = DataProfiler(self.df)
            cleaned_profile = cleaned_profiler.profile_dataset()
            
            # Save cleaned profile report
            if self.config.output.report_format == 'html':
                cleaned_profile_file = os.path.join(output_dir, f"{base_name}_cleaned_profile_{timestamp}.html")
                cleaned_profiler.generate_profile_report(cleaned_profile_file)
            else:
                cleaned_profile_file = os.path.join(output_dir, f"{base_name}_cleaned_profile_{timestamp}.json")
                with open(cleaned_profile_file, 'w') as f:
                    json.dump(cleaned_profile, f, indent=2)
                    
            logger.info(f"Saved cleaned data profile to {cleaned_profile_file}")
            
            # 3. Generate a comparison report
            if self.original_df is not None:
                self._generate_comparison_report(
                    self.original_df, 
                    self.df, 
                    os.path.join(output_dir, f"{base_name}_comparison_{timestamp}.html")
                )
    
    def _generate_comparison_report(self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame, output_path: str) -> None:
        """Generate a report comparing original and cleaned data"""
        try:
            import jinja2
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            
            # Create comparison statistics
            comparison = {
                'original_rows': len(original_df),
                'cleaned_rows': len(cleaned_df),
                'rows_difference': len(original_df) - len(cleaned_df),
                'rows_difference_pct': (len(original_df) - len(cleaned_df)) / len(original_df) * 100 if len(original_df) > 0 else 0,
                
                'original_columns': len(original_df.columns),
                'cleaned_columns': len(cleaned_df.columns),
                'columns_difference': len(original_df.columns) - len(cleaned_df.columns),
                
                'original_memory_mb': original_df.memory_usage(deep=True).sum() / (1024 * 1024),
                'cleaned_memory_mb': cleaned_df.memory_usage(deep=True).sum() / (1024 * 1024),
                'memory_difference_pct': (1 - cleaned_df.memory_usage(deep=True).sum() / original_df.memory_usage(deep=True).sum()) * 100 if original_df.memory_usage(deep=True).sum() > 0 else 0,
                
                'original_dtypes': original_df.dtypes.value_counts().to_dict(),
                'cleaned_dtypes': cleaned_df.dtypes.value_counts().to_dict(),
                
                'original_missing_total': original_df.isna().sum().sum(),
                'cleaned_missing_total': cleaned_df.isna().sum().sum(),
                'missing_difference': original_df.isna().sum().sum() - cleaned_df.isna().sum().sum(),
                'missing_difference_pct': (original_df.isna().sum().sum() - cleaned_df.isna().sum().sum()) / max(1, original_df.isna().sum().sum()) * 100,
                
                'original_duplicate_rows': original_df.duplicated().sum(),
                'cleaned_duplicate_rows': cleaned_df.duplicated().sum(),
                
                'column_comparison': []
            }
            
            # Compare common columns
            common_columns = set(original_df.columns) & set(cleaned_df.columns)
            
            for col in common_columns:
                # Skip if too many values in either dataframe
                if len(original_df) > 10000 or len(cleaned_df) > 10000:
                    continue
                    
                try:
                    # Create a chart comparing distributions for numeric columns
                    if pd.api.types.is_numeric_dtype(original_df[col]) and pd.api.types.is_numeric_dtype(cleaned_df[col]):
                        # Generate histograms
                        plt.figure(figsize=(10, 6))
                        plt.hist(original_df[col].dropna(), alpha=0.5, bins=30, label='Original')
                        plt.hist(cleaned_df[col].dropna(), alpha=0.5, bins=30, label='Cleaned')
                        plt.legend()
                        plt.title(f'Distribution of {col}')
                        plt.tight_layout()
                        
                        # Save to file
                        chart_filename = f"chart_{col}_{hashlib.md5(col.encode()).hexdigest()[:6]}.png"
                        chart_path = os.path.join(os.path.dirname(output_path), chart_filename)
                        plt.savefig(chart_path)
                        plt.close()
                        
                        # Add to comparison
                        comparison['column_comparison'].append({
                            'column': col,
                            'chart': chart_filename,
                            'original_dtype': str(original_df[col].dtype),
                            'cleaned_dtype': str(cleaned_df[col].dtype),
                            'original_missing': int(original_df[col].isna().sum()),
                            'cleaned_missing': int(cleaned_df[col].isna().sum()),
                            'original_unique': int(original_df[col].nunique()),
                            'cleaned_unique': int(cleaned_df[col].nunique()),
                        })
                except:
                    continue
            
            # Create HTML template
            template_str = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Data Cleaning Comparison Report</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    h1, h2, h3 { color: #333366; }
                    table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                    th, td { padding: 8px; text-align: left; border: 1px solid #ddd; }
                    th { background-color: #f2f2f2; }
                    .section { margin-bottom: 30px; border: 1px solid #ddd; padding: 20px; border-radius: 5px; }
                    .chart { max-width: 100%; height: auto; }
                    .good { color: green; }
                    .bad { color: red; }
                    .metrics { display: flex; flex-wrap: wrap; }
                    .metric-box { 
                        width: 200px; 
                        height: 120px;
                        margin: 10px;
                        padding: 15px;
                        border-radius: 5px;
                        background-color: #f8f9fa;
                        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                        display: flex;
                        flex-direction: column;
                        justify-content: center;
                    }
                    .metric-title { font-size: 16px; color: #666; margin-bottom: 10px; }
                    .metric-value { font-size: 24px; font-weight: bold; color: #333; }
                    .metric-change { font-size: 14px; margin-top: 5px; }
                </style>
            </head>
            <body>
                <h1>Data Cleaning Comparison Report</h1>
                <p>Generated on {{ now }}</p>
                
                <div class="section">
                    <h2>Summary</h2>
                    <div class="metrics">
                        <div class="metric-box">
                            <div class="metric-title">Rows</div>
                            <div class="metric-value">{{ comparison.cleaned_rows }}</div>
                            <div class="metric-change {% if comparison.rows_difference > 0 %}bad{% else %}good{% endif %}">
                                {{ comparison.rows_difference_pct|abs|round(1) }}% {% if comparison.rows_difference > 0 %}less{% else %}more{% endif %}
                            </div>
                        </div>
                        
                        <div class="metric-box">
                            <div class="metric-title">Columns</div>
                            <div class="metric-value">{{ comparison.cleaned_columns }}</div>
                            <div class="metric-change">
                                {% if comparison.columns_difference != 0 %}
                                    {{ comparison.columns_difference|abs }} {% if comparison.columns_difference > 0 %}less{% else %}more{% endif %}
                                {% else %}
                                    unchanged
                                {% endif %}
                            </div>
                        </div>
                        
                        <div class="metric-box">
                            <div class="metric-title">Missing Values</div>
                            <div class="metric-value">{{ comparison.cleaned_missing_total }}</div>
                            <div class="metric-change {% if comparison.missing_difference > 0 %}good{% else %}bad{% endif %}">
                                {{ comparison.missing_difference_pct|abs|round(1) }}% {% if comparison.missing_difference > 0 %}less{% else %}more{% endif %}
                            </div>
                        </div>
                        
                        <div class="metric-box">
                            <div class="metric-title">Memory Usage</div>
                            <div class="metric-value">{{ comparison.cleaned_memory_mb|round(1) }} MB</div>
                            <div class="metric-change {% if comparison.memory_difference_pct > 0 %}good{% else %}bad{% endif %}">
                                {{ comparison.memory_difference_pct|abs|round(1) }}% {% if comparison.memory_difference_pct > 0 %}less{% else %}more{% endif %}
                            </div>
                        </div>
                        
                        <div class="metric-box">
                            <div class="metric-title">Duplicate Rows</div>
                            <div class="metric-value">{{ comparison.cleaned_duplicate_rows }}</div>
                            <div class="metric-change {% if comparison.original_duplicate_rows > comparison.cleaned_duplicate_rows %}good{% elif comparison.original_duplicate_rows < comparison.cleaned_duplicate_rows %}bad{% endif %}">
                                {% if comparison.original_duplicate_rows != comparison.cleaned_duplicate_rows %}
                                    {{ (comparison.original_duplicate_rows - comparison.cleaned_duplicate_rows)|abs }} {% if comparison.original_duplicate_rows > comparison.cleaned_duplicate_rows %}less{% else %}more{% endif %}
                                {% else %}
                                    unchanged
                                {% endif %}
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>Data Types</h2>
                    <div style="display: flex; justify-content: space-between;">
                        <div style="flex: 1;">
                            <h3>Original Data Types</h3>
                            <table>
                                <tr><th>Type</th><th>Count</th></tr>
                                {% for dtype, count in comparison.original_dtypes.items() %}
                                <tr><td>{{ dtype }}</td><td>{{ count }}</td></tr>
                                {% endfor %}
                            </table>
                        </div>
                        <div style="flex: 1;">
                            <h3>Cleaned Data Types</h3>
                            <table>
                                <tr><th>Type</th><th>Count</th></tr>
                                {% for dtype, count in comparison.cleaned_dtypes.items() %}
                                <tr><td>{{ dtype }}</td><td>{{ count }}</td></tr>
                                {% endfor %}
                            </table>
                        </div>
                    </div>
                </div>
                
                {% if comparison.column_comparison %}
                <div class="section">
                    <h2>Column Comparisons</h2>
                    {% for col_comp in comparison.column_comparison %}
                    <div style="margin-bottom: 30px;">
                        <h3>{{ col_comp.column }}</h3>
                        <div style="display: flex; flex-wrap: wrap;">
                            <div style="flex: 1; min-width: 300px;">
                                <table>
                                    <tr>
                                        <th>Metric</th>
                                        <th>Original</th>
                                        <th>Cleaned</th>
                                        <th>Change</th>
                                    </tr>
                                    <tr>
                                        <td>Data Type</td>
                                        <td>{{ col_comp.original_dtype }}</td>
                                        <td>{{ col_comp.cleaned_dtype }}</td>
                                        <td>
                                            {% if col_comp.original_dtype != col_comp.cleaned_dtype %}
                                                Changed
                                            {% else %}
                                                Unchanged
                                            {% endif %}
                                        </td>
                                    </tr>
                                    <tr>
                                        <td>Missing Values</td>
                                        <td>{{ col_comp.original_missing }}</td>
                                        <td>{{ col_comp.cleaned_missing }}</td>
                                        <td>
                                            {% if col_comp.original_missing > col_comp.cleaned_missing %}
                                                <span class="good">-{{ col_comp.original_missing - col_comp.cleaned_missing }}</span>
                                            {% elif col_comp.original_missing < col_comp.cleaned_missing %}
                                                <span class="bad">+{{ col_comp.cleaned_missing - col_comp.original_missing }}</span>
                                            {% else %}
                                                Unchanged
                                            {% endif %}
                                        </td>
                                    </tr>
                                    <tr>
                                        <td>Unique Values</td>
                                        <td>{{ col_comp.original_unique }}</td>
                                        <td>{{ col_comp.cleaned_unique }}</td>
                                        <td>
                                            {% if col_comp.original_unique != col_comp.cleaned_unique %}
                                                {{ col_comp.cleaned_unique - col_comp.original_unique }}
                                            {% else %}
                                                Unchanged
                                            {% endif %}
                                        </td>
                                    </tr>
                                </table>
                            </div>
                            <div style="flex: 1; min-width: 300px; text-align: center;">
                                {% if col_comp.chart %}
                                <img src="{{ col_comp.chart }}" alt="Distribution Chart" class="chart">
                                {% endif %}
                            </div>
                        </div>
                    </div>
                    {% endfor %}
                </div>
                {% endif %}
                
                <div class="footer">
                    <p>Generated using Advanced Data Cleaner</p>
                </div>
            </body>
            </html>
            """
            
            # Create Jinja template
            template = jinja2.Template(template_str)
            
            # Render template with comparison data
            html = template.render(
                comparison=comparison, 
                now=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            )
            
            # Write to file
            with open(output_path, 'w') as f:
                f.write(html)
                
            logger.info(f"Comparison report written to {output_path}")
            
        except Exception as e:
            logger.error(f"Error generating comparison report: {str(e)}")
            
            # Fallback to JSON report
            comparison_file = os.path.join(os.path.dirname(output_path), f"comparison_{int(time.time())}.json")
            try:
                with open(comparison_file, 'w') as f:
                    json.dump(self.stats, f, indent=2, default=str)
                logger.info(f"Fallback comparison JSON written to {comparison_file}")
            except:
                logger.error("Failed to write fallback comparison report")


#################################################
# Command Line Interface
#################################################

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Advanced Data Cleaner - Enterprise-grade data cleaning tool')
    
    # Input/output options
    parser.add_argument('--input', '-i', required=True, help='Input CSV file path')
    parser.add_argument('--output', '-o', help='Output file path (default: input filename with _cleaned suffix)')
    parser.add_argument('--config', '-c', help='Path to configuration file (YAML or JSON)')
    parser.add_argument('--delimiter', '-d', help='CSV delimiter')
    parser.add_argument('--encoding', '-e', help='File encoding')
    
    # Pipeline control
    parser.add_argument('--pipeline', help='Comma-separated list of pipeline steps to run')
    parser.add_argument('--skip-steps', help='Comma-separated list of pipeline steps to skip')
    
    # Column filtering
    parser.add_argument('--keep-columns', help='Comma-separated list of columns to keep')
    parser.add_argument('--drop-columns', help='Comma-separated list of columns to drop')
    
    # Missing value handling
    parser.add_argument('--missing-strategy', choices=['auto', 'drop', 'impute', 'flag', 'none'], 
                        help='Strategy for handling missing values')
    parser.add_argument('--missing-threshold-row', type=float, 
                        help='Threshold for dropping rows with missing values (0-1)')
    parser.add_argument('--missing-threshold-col', type=float, 
                        help='Threshold for dropping columns with missing values (0-1)')
    
    # Outlier handling
    parser.add_argument('--outlier-method', choices=['zscore', 'iqr', 'isolation_forest', 'none', 'auto'], 
                        help='Method for detecting outliers')
    parser.add_argument('--outlier-action', choices=['remove', 'cap', 'flag', 'none', 'auto'], 
                        help='Action to take with outliers')
    
    # Duplicate handling
    parser.add_argument('--duplicate-strategy', choices=['remove_first', 'remove_last', 'remove_all', 'flag', 'none', 'auto'], 
                        help='Strategy for handling duplicate rows')
    
    # Output options
    parser.add_argument('--report', action='store_true', help='Generate data quality reports')
    parser.add_argument('--report-format', choices=['html', 'json', 'txt'], help='Format for reports')
    parser.add_argument('--save-intermediate', action='store_true', help='Save intermediate results')
    parser.add_argument('--compress', action='store_true', help='Compress output files')
    parser.add_argument('--output-format', help='Output file format (csv, parquet, etc.)')
    
    # Performance options
    parser.add_argument('--chunk-size', type=int, help='Chunk size for processing large files')
    parser.add_argument('--jobs', '-j', type=int, help='Number of parallel jobs')
    
    # Misc
    parser.add_argument('--verbose', '-v', action='count', default=0, help='Increase verbosity (can be used multiple times)')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress all output except errors')
    parser.add_argument('--version', action='version', version='Advanced Data Cleaner 1.0.0')
    parser.add_argument('--generate-config', help='Generate a default configuration file at the specified path and exit')
    
    return parser.parse_args()


def configure_from_args(cleaner: DataCleaner, args) -> None:
    """Configure DataCleaner from command line arguments"""
    # Input/output options
    cleaner.config.input_file = args.input
    if args.output:
        cleaner.config.output_file = args.output
        
    if args.delimiter:
        cleaner.config.delimiter = args.delimiter
        
    if args.encoding:
        cleaner.config.encoding = args.encoding
    
    # Pipeline control
    if args.pipeline:
        cleaner.config.pipeline_steps = args.pipeline.split(',')
        
    if args.skip_steps:
        skip_steps = args.skip_steps.split(',')
        cleaner.config.pipeline_steps = [step for step in cleaner.config.pipeline_steps if step not in skip_steps]
    
    # Column filtering
    if args.keep_columns:
        cleaner.config.columns_to_keep = args.keep_columns.split(',')
        
    if args.drop_columns:
        cleaner.config.columns_to_drop = args.drop_columns.split(',')
    
    # Missing value handling
    if args.missing_strategy:
        cleaner.config.missing.strategy = args.missing_strategy
        
    if args.missing_threshold_row:
        cleaner.config.missing.threshold_row = args.missing_threshold_row
        
    if args.missing_threshold_col:
        cleaner.config.missing.threshold_column = args.missing_threshold_col
    
    # Outlier handling
    if args.outlier_method:
        cleaner.config.outliers.detection_method = args.outlier_method
        
    if args.outlier_action:
        cleaner.config.outliers.handling_strategy = args.outlier_action
    
    # Duplicate handling
    if args.duplicate_strategy:
        cleaner.config.duplicates.strategy = args.duplicate_strategy
    
    # Output options
    if args.report:
        cleaner.config.output.create_report = True
        
    if args.report_format:
        cleaner.config.output.report_format = args.report_format
        
    if args.save_intermediate:
        cleaner.config.output.save_intermediate = True
        
    if args.compress:
        cleaner.config.output.compress_output = True
        
    if args.output_format:
        cleaner.config.output.output_format = args.output_format
    
    # Performance options
    if args.chunk_size:
        cleaner.config.performance.chunk_size = args.chunk_size
        
    if args.jobs:
        cleaner.config.performance.n_jobs = args.jobs
    
    # Logging level
    if args.quiet:
        logger.setLevel(logging.ERROR)
    elif args.verbose > 1:
        logger.setLevel(logging.DEBUG)
    elif args.verbose == 1:
        logger.setLevel(logging.INFO)


def generate_default_config(output_path: str) -> None:
    """Generate a default configuration file"""
    config = CleanerConfig()
    
    # Convert to dict
    config_dict = {}
    for key, value in config.__dict__.items():
        if isinstance(value, (MissingValueConfig, OutlierConfig, DataTypeConfig,
                             DuplicateConfig, NormalizationConfig, EncodingConfig,
                             ValidationConfig, PerformanceConfig, OutputConfig)):
            # Handle nested configs
            config_dict[key] = {}
            for subkey, subvalue in value.__dict__.items():
                if not isinstance(subvalue, (list, dict)) or subvalue:
                    config_dict[key][subkey] = subvalue
        else:
            if not isinstance(value, (list, dict)) or value:
                config_dict[key] = value
    
    # Write to file
    file_ext = os.path.splitext(output_path)[1].lower()
    
    if file_ext == '.yaml' or file_ext == '.yml':
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    elif file_ext == '.json':
        with open(output_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    else:
        # Default to YAML
        output_path = f"{output_path}.yaml"
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    print(f"Generated default configuration at {output_path}")


def main():
    """Main entry point for command-line execution"""
    # Parse arguments
    args = parse_args()
    
    # Handle generate-config option
    if args.generate_config:
        generate_default_config(args.generate_config)
        return 0
    
    # Create cleaner instance
    cleaner = DataCleaner()
    
    # Load config file if provided
    if args.config:
        try:
            cleaner.load_config(args.config)
        except Exception as e:
            logger.error(f"Error loading config file: {str(e)}")
            return 1
    
    # Apply command line arguments
    configure_from_args(cleaner, args)
    
    try:
        # Execute pipeline
        start_time = time.time()
        
        # Read data
        df = cleaner.read_data()
        
        # Clean data
        df = cleaner.clean_data(df)
        
        # Write output
        cleaner.write_data()
        
        total_time = time.time() - start_time
        logger.info(f"Total processing time: {total_time:.2f} seconds")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
    