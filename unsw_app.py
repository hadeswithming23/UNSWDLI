from cProfile import label
import streamlit as st
import pandas as pd
import joblib

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import io

import warnings

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Intrusion Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .normal {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .attack {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load trained model and preprocessing objects."""
    models_dict = {}
    try:
        # Core model
        models_dict['xgb'] = joblib.load("unsw_rf_full.pkl")
        models_dict["encoders"] = joblib.load("unsw_encoders.pkl")

        # Preprocessing
        st.success("✅ All models and preprocessing objects loaded successfully!")

    except Exception as e:
        st.error(f"❌ Error loading models: {e}")

    return models_dict



@st.cache_data
def get_feature_names():
    """Get the feature names used in training"""
    # Based on the notebook analysis, these are the top features
    return  ['dur', 'state', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 'sload', 'dload', 'sinpkt', 'dinpkt', 'tcprtt', 'synack', 'ackdat', 'smean', 'dmean', 'ct_srv_src', 'ct_state_ttl', 'ct_dst_src_ltm', 'ct_srv_dst']



@st.cache_data
def get_complete_column_names():
    """Get the complete column names for NSL-KDD dataset"""
    return [
        "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
        "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
        "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
        "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
        "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
        "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
        "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
        "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
        "dst_host_serror_rate", "dst_host_srv_serror_rate",
        "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
        "attack", "level"
    ]

@st.cache_data
def get_categorical_options():
    """Get categorical feature options"""
    return {
        'protocol_type': ['tcp', 'udp', 'icmp'],
        'service': ['http', 'smtp', 'finger', 'auth', 'telnet', 'ftp', 'private', 'pop_3', 'ftp_data', 'ntp_u', 'other', 'ecr_i', 'time', 'domain', 'ssh', 'name', 'whois', 'mtp', 'gopher', 'rje', 'vmnet', 'daytime', 'link', 'supdup', 'uucp', 'netstat', 'kshell', 'echo', 'discard', 'systat', 'csnet_ns', 'iso_tsap', 'hostnames', 'exec', 'login', 'shell', 'printer', 'efs', 'courier', 'uucp_path', 'netbios_ns', 'netbios_dgm', 'netbios_ssn', 'sql_net', 'X11', 'urh_i', 'urp_i', 'pm_dump', 'tftp_u', 'red_i', 'harvest'],
        'flag': ['SF', 'S0', 'REJ', 'RSTR', 'RSTO', 'S1', 'RSTOS0', 'S3', 'S2', 'OTH', 'SH']
    }

# Define UNSW columns
UNSW_COLUMNS = [
    "srcip", "sport", "dstip", "dsport", "proto", "state", "dur", "sbytes", "dbytes",
    "sttl", "dttl", "sloss", "dloss", "service", "Sload", "Dload", "Spkts", "Dpkts",
    "swin", "dwin", "stcpb", "dtcpb", "smeansz", "dmeansz", "trans_depth", "res_bdy_len",
    "Sjit", "Djit", "Stime", "Ltime", "Sintpkt", "Dintpkt", "tcprtt", "synack", "ackdat",
    "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd",
    "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm",
    "ct_dst_sport_ltm", "ct_dst_src_ltm", "attack_cat", "Label"
]

UNSW_SELECTED_COLUMNS = ['dur', 'proto', 'service', 'state', 'spkts', 'dpkts', 'sbytes',
       'dbytes', 'rate', 'sttl', 'dttl', 'sload', 'dload', 'sloss', 'dloss',
       'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin',
       'tcprtt', 'synack', 'ackdat', 'smean', 'dmean', 'trans_depth',
       'response_body_len', 'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm',
       'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm',
       'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm',
       'ct_srv_dst', 'is_sm_ips_ports']

# Define NSL-KDD columns (shortened example — expand with your full schema)
NSLKDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "attack", "level"
]


def map_unsw_to_nsl(unsw_row: dict) -> dict:
    """Map UNSW-NB15 row to NSL-KDD style features for model compatibility."""
    # Defensive defaults
    ct_srv_src = unsw_row.get("ct_srv_src", 0)
    ct_srv_dst = unsw_row.get("ct_srv_dst", 0)
    ct_dst_ltm = unsw_row.get("ct_dst_ltm", 1)
    ct_src_dport_ltm = unsw_row.get("ct_src_dport_ltm", 1)
    ct_dst_src_ltm = unsw_row.get("ct_dst_src_ltm", 1)
    ct_state_ttl = unsw_row.get("ct_state_ttl", 0)

    # Map UNSW state to NSL flag
    state_map = {
        "CON": "SF",   # connection established
        "FIN": "SF",   # finished normally
        "INT": "S0",   # attempt, no response
        "RST": "REJ",  # reset
        "ECO": "S1",   # echo request
        "PAR": "RSTR", # partial reset
        "REQ": "S2",   # other request
        "ACC": "SF",   # accepted
    }
    flag = state_map.get(unsw_row.get("state", "CON"), "SF")

    mapped = {
        "duration": unsw_row.get("dur", 0),
        "src_bytes": unsw_row.get("sbytes", 0),
        "dst_bytes": unsw_row.get("dbytes", 0),
        "protocol_type": unsw_row.get("proto", "tcp"),
        "service": unsw_row.get("service", "-"),
        "flag": flag,
        "logged_in": unsw_row.get("is_ftp_login", 0),
        "count": ct_srv_src,
        "srv_count": ct_srv_dst,
        "serror_rate": ct_state_ttl / max(ct_srv_src + ct_srv_dst, 1),
        "srv_serror_rate": ct_state_ttl / max(ct_srv_dst, 1),
        "rerror_rate": ct_state_ttl / max(ct_srv_src, 1),
        "srv_rerror_rate": ct_state_ttl / max(ct_srv_dst, 1),
        "same_srv_rate": ct_srv_dst / max(ct_dst_ltm, 1),
        "diff_srv_rate": (ct_dst_ltm - ct_srv_dst) / max(ct_dst_ltm, 1),
        "srv_diff_host_rate": (ct_src_dport_ltm - ct_srv_dst) / max(ct_src_dport_ltm, 1),
        "dst_host_count": ct_dst_ltm,
        "dst_host_srv_count": ct_srv_dst,
        "dst_host_same_srv_rate": ct_srv_dst / max(ct_dst_ltm, 1),
        "dst_host_diff_srv_rate": (ct_dst_ltm - ct_srv_dst) / max(ct_dst_ltm, 1),
        "dst_host_same_src_port_rate": ct_src_dport_ltm / max(ct_dst_ltm, 1),
        "dst_host_srv_diff_host_rate": (ct_dst_src_ltm - ct_srv_dst) / max(ct_dst_src_ltm, 1),
        "dst_host_serror_rate": ct_state_ttl / max(ct_dst_ltm, 1),
        "dst_host_srv_serror_rate": ct_state_ttl / max(ct_srv_dst, 1),
        "dst_host_rerror_rate": ct_state_ttl / max(ct_dst_ltm, 1),
        "dst_host_srv_rerror_rate": ct_state_ttl / max(ct_srv_dst, 1),
        "level": unsw_row.get("label", 0),
    }
    return mapped


import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

def preprocess_input(df_input, encoders, label_col):
    """
    Preprocess input (DataFrame or dict) for UNSW-NB15 with full feature set.

    Steps:
    1. Convert dict to DataFrame
    2. Clean empty strings and invalid values
    3. Separate label column if exists
    4. Drop unnecessary columns: 'id', 'attack_cat'
    5. Encode categorical features using saved LabelEncoders
    6. Add missing features with zeros, drop extras
    7. Handle NA values and ensure numeric output
    8. Return aligned features and optional labels
    """

    import pandas as pd
    import numpy as np

    # --- Step 1: Convert dict to DataFrame ---
    if isinstance(df_input, dict):
        df_input = pd.DataFrame([df_input])

    # --- Step 2: Clean empty strings and invalid values ---
    # Replace empty strings or whitespace with NaN
    df_input = df_input.replace(r'^\s*$', np.nan, regex=True)

    # --- Step 3: Separate label if exists ---
    y_true = None
    if label_col in df_input.columns:
        y_true = df_input[label_col].apply(lambda x: 0 if x == 0 else 1)
        df_input = df_input.drop(columns=[label_col])

    # --- Step 4: Drop unnecessary columns ---
    drop_cols = ["id", "attack_cat"]
    df_input = df_input.drop(columns=[c for c in drop_cols if c in df_input.columns], errors="ignore")

    # --- Step 5: Encode categorical features ---
    cat_cols = ["proto", "service", "state"]
    for col in cat_cols:
        if col in df_input.columns and col in encoders:
            le = encoders[col]
            # Handle unknown categories and ensure numeric output
            df_input[col] = df_input[col].apply(
                lambda s: le.transform([s])[0] if isinstance(s, str) and s in le.classes_ else -1
            ).astype(float)

    # --- Step 6: Add missing features with 0 & drop extras ---
    for col in UNSW_SELECTED_COLUMNS:
        if col not in df_input.columns:
            df_input[col] = 0
    df_input = df_input[UNSW_SELECTED_COLUMNS]  # Enforce column order

    # --- Step 7: Handle NA values and ensure numeric ---
    df_input = df_input.fillna(0)
    # Convert all columns to float to ensure numeric output
    try:
        df_input = df_input.astype(float)
    except ValueError as e:
        print(f"Non-numeric values found in columns: {e}")
        for col in df_input.columns:
            if df_input[col].dtype not in [np.float64, np.int64]:
                print(f"Column {col} has non-numeric values:\n", df_input[col].unique())
        raise

    # --- Step 8: Validate output ---
    X = df_input.values
    print("Final X shape:", X.shape, "dtype:", X.dtype)  # Debug
    if not np.issubdtype(X.dtype, np.number):
        raise ValueError("Output contains non-numeric values")

    return X, y_true


from sklearn.metrics import accuracy_score
import pandas as pd
import joblib

def predict_attack(models: dict, X_scaled, y_true=None):
    import pandas as pd
    import numpy as np
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    if isinstance(X_scaled, pd.DataFrame):
        X_scaled = X_scaled.values

    xgb_model = models["xgb"]

    # Predict numeric classes
    preds = xgb_model.predict(X_scaled).astype(int)

    # Flip 0 ↔ 1 if model is inverted
    # preds_flipped = 1 - preds.astype(int)  

    probs = xgb_model.predict_proba(X_scaled)

    results_list = []
    for i in range(len(preds)):
        results_list.append({
            "index": i,
            "prediction": int(preds[i]),  # 0=normal, 1=attack after flip
            "normal_prob": float(probs[i][0]),    # swap prob columns if needed
            "attack_prob": float(probs[i][1]),
            "status": "Normal" if preds[i] == 0 else "Attack"
        })

    results_df = pd.DataFrame(results_list)

    metrics = None
    if y_true is not None:
        y_true_arr = np.array(y_true)
        y_pred_labels = preds
        metrics = {
            "accuracy": accuracy_score(y_true_arr, y_pred_labels),
            "precision": precision_score(y_true_arr, y_pred_labels, zero_division=0),
            "recall": recall_score(y_true_arr, y_pred_labels, zero_division=0),
            "f1": f1_score(y_true_arr, y_pred_labels, zero_division=0)
        }

    return results_df, metrics


def main():
    # Header
    st.markdown('<h1 class="main-header">🛡️ Intrusion Detection System</h1>', unsafe_allow_html=True)
    
    # Load model
    with st.spinner("Loading ensemble model..."):
        models = load_model()  # dictionary of models
    
    if models is None or len(models) == 0:
        st.error("Failed to load model. Please check if the model files are in the correct location.")
        return
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Batch Upload", "Model Information", "About"]
    )
    
    if page == "Batch Upload":
        batch_upload_page(models)  # pass the dictionary
    elif page == "Model Information":
        model_info_page(models)
    elif page == "About":
        about_page()


def prediction_page(models):
    """Main prediction interface"""
    st.header("🔍 Network Traffic Analysis")
    st.markdown("Enter network traffic features to detect potential intrusions.")
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Basic Connection Features")
        
        # Basic features
        duration = st.number_input("Duration (seconds)", min_value=0, max_value=100000, value=0)
        protocol_type = st.selectbox("Protocol Type", get_categorical_options()['protocol_type'])
        service = st.selectbox("Service", get_categorical_options()['service'])
        flag = st.selectbox("Flag", get_categorical_options()['flag'])
        
        src_bytes = st.number_input("Source Bytes", min_value=0, max_value=1000000000, value=0)
        dst_bytes = st.number_input("Destination Bytes", min_value=0, max_value=1000000000, value=0)
        
        # Connection features
        land = st.selectbox("Land (same host/port)", [0, 1])
        wrong_fragment = st.number_input("Wrong Fragment", min_value=0, max_value=100, value=0)
        urgent = st.number_input("Urgent", min_value=0, max_value=100, value=0)
        hot = st.number_input("Hot", min_value=0, max_value=100, value=0)
        
    with col2:
        st.subheader("Advanced Features")
        
        # Authentication features
        num_failed_logins = st.number_input("Number of Failed Logins", min_value=0, max_value=100, value=0)
        logged_in = st.selectbox("Logged In", [0, 1])
        num_compromised = st.number_input("Number Compromised", min_value=0, max_value=100, value=0)
        root_shell = st.selectbox("Root Shell", [0, 1])
        su_attempted = st.selectbox("SU Attempted", [0, 1])
        
        # File and shell features
        num_root = st.number_input("Number of Root Accesses", min_value=0, max_value=100, value=0)
        num_file_creations = st.number_input("Number of File Creations", min_value=0, max_value=100, value=0)
        num_shells = st.number_input("Number of Shells", min_value=0, max_value=100, value=0)
        num_access_files = st.number_input("Number of Access Files", min_value=0, max_value=100, value=0)
        
    # Additional features in a third column
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Connection Statistics")
        
        count = st.number_input("Count", min_value=0, max_value=1000, value=0)
        srv_count = st.number_input("Service Count", min_value=0, max_value=1000, value=0)
        serror_rate = st.slider("Service Error Rate", 0.0, 1.0, 0.0, 0.01)
        srv_serror_rate = st.slider("Service Service Error Rate", 0.0, 1.0, 0.0, 0.01)
        rerror_rate = st.slider("Response Error Rate", 0.0, 1.0, 0.0, 0.01)
        srv_rerror_rate = st.slider("Service Response Error Rate", 0.0, 1.0, 0.0, 0.01)
        
    with col4:
        st.subheader("Host Statistics")
        
        same_srv_rate = st.slider("Same Service Rate", 0.0, 1.0, 0.0, 0.01)
        diff_srv_rate = st.slider("Different Service Rate", 0.0, 1.0, 0.0, 0.01)
        dst_host_count = st.number_input("Destination Host Count", min_value=0, max_value=1000, value=0)
        dst_host_srv_count = st.number_input("Destination Host Service Count", min_value=0, max_value=1000, value=0)
        dst_host_same_srv_rate = st.slider("Destination Host Same Service Rate", 0.0, 1.0, 0.0, 0.01)
        dst_host_diff_srv_rate = st.slider("Destination Host Different Service Rate", 0.0, 1.0, 0.0, 0.01)
    
    # Prediction button
    st.markdown("---")
    if st.button("🔍 Analyze Traffic", type="primary", use_container_width=True):
        with st.spinner("Analyzing network traffic..."):
            # Collect all input data
            input_data = {
                'duration': duration,
                'protocol_type': protocol_type,
                'service': service,
                'flag': flag,
                'src_bytes': src_bytes,
                'dst_bytes': dst_bytes,
                'land': land,
                'wrong_fragment': wrong_fragment,
                'urgent': urgent,
                'hot': hot,
                'num_failed_logins': num_failed_logins,
                'logged_in': logged_in,
                'num_compromised': num_compromised,
                'root_shell': root_shell,
                'su_attempted': su_attempted,
                'num_root': num_root,
                'num_file_creations': num_file_creations,
                'num_shells': num_shells,
                'num_access_files': num_access_files,
                'count': count,
                'srv_count': srv_count,
                'serror_rate': serror_rate,
                'srv_serror_rate': srv_serror_rate,
                'rerror_rate': rerror_rate,
                'srv_rerror_rate': srv_rerror_rate,
                'same_srv_rate': same_srv_rate,
                'diff_srv_rate': diff_srv_rate,
                'dst_host_count': dst_host_count,
                'dst_host_srv_count': dst_host_srv_count,
                'dst_host_same_srv_rate': dst_host_same_srv_rate,
                'dst_host_diff_srv_rate': dst_host_diff_srv_rate
            }
            

            # Then call predict_attack correctly
            result = predict_attack(models, input_data)
            
            if result:
                # Display results
                st.markdown("## 📊 Analysis Results")
                
                # Prediction result
                prediction = result['prediction']
                probability = result['probability']
                
                if prediction == 0:
                    st.markdown(
                        f'<div class="prediction-box normal">'
                        f'<h3>✅ Normal Traffic Detected</h3>'
                        f'<p>Confidence: {probability[0]:.2%}</p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="prediction-box attack">'
                        f'<h3>🚨 Attack Detected!</h3>'
                        f'<p>Confidence: {probability[1]:.2%}</p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                
                # Display probability breakdown
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Normal Probability", f"{probability[0]:.2%}")
                with col2:
                    st.metric("Attack Probability", f"{probability[1]:.2%}")
                
                # Feature importance visualization
                st.subheader("Feature Analysis")
                processed_features = result['processed_features']
                feature_importance = processed_features.iloc[0].abs().sort_values(ascending=False)
                
                st.bar_chart(feature_importance.head(10))

def batch_upload_page(models):
    """Batch upload and prediction interface (no apply_feature_mapping)"""
    import pandas as pd
    import numpy as np
    import re
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    st.header("📁 Batch File Upload")
    st.markdown("Upload a CSV or TXT file with network traffic data for batch analysis.")

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'txt'],
        help="Upload a CSV or TXT file with network traffic data"
    )

    
    if uploaded_file is not None:
        file_name = uploaded_file.name.lower()
        
        # Determine default columns based on file name
        if "unsw" in file_name:
            default_cols = UNSW_COLUMNS
        elif "kdd" in file_name or "nsl" in file_name:
            default_cols = NSLKDD_COLUMNS
        else:
            default_cols = None

        # Read file content into memory
        content = uploaded_file.read()
        buffer = io.BytesIO(content)

        try:
            # Step 1: Preview first 5 rows to detect headers
            preview_df = pd.read_csv(io.BytesIO(content), nrows=5, header=None)
            has_headers = all(isinstance(x, str) for x in preview_df.iloc[0].values)

            # Step 2: Load full DataFrame based on header detection
            buffer.seek(0)
            if has_headers:
                df_orig = pd.read_csv(buffer)
                st.info("✅ Headers detected — using provided headers.")
            else:
                df_orig = pd.read_csv(buffer, header=None)
                if default_cols and len(default_cols) == df_orig.shape[1]:
                    df_orig.columns = default_cols
                    st.info("ℹ️ No headers detected — assigned default columns from file type.")
                else:
                    df_orig.columns = [f"col_{i}" for i in range(df_orig.shape[1])]
                    st.warning("⚠️ No headers detected — assigned generic column names.")

            # Step 3: Show a preview of the DataFrame
            st.success(f"✅ File uploaded successfully! Shape: {df_orig.shape}")
            st.subheader("📊 Data Preview")
            st.dataframe(df_orig.head(10), use_container_width=True)

        except Exception as e:
            st.error(f"❌ Failed to read uploaded file: {e}")


        # -----------------------
        # 2) Dataset detection
        # -----------------------
        if "state" in df_orig.columns or "ct_srv_src" in df_orig.columns:
            dataset_type = "UNSW"
        elif "flag" in df_orig.columns or "attack" in df_orig.columns:
            dataset_type = "NSL"
        else:
            dataset_type = "Unknown"
        st.info(f"Detected dataset type: **{dataset_type}**")

        # -----------------------
        # 3) Label column choice
        # -----------------------
        st.subheader("🏷️ Label Column (optional)")
        label_col = st.selectbox(
            "Select label column (if available)",
            options=["<None>"] + list(df_orig.columns),
            index=0
        )
        if label_col == "<None>":
            label_col = None
        if label_col:
            st.success(f"✅ Label column selected: {label_col}")

        # -----------------------
        # 4) Feature alignment
        # -----------------------
        expected_features = UNSW_SELECTED_COLUMNS

        if dataset_type == "UNSW":
            # Preprocess UNSW-NB15 dataset only
            df_mapped = df_orig.copy()  # just work with a copy of the original dataframe

            # Ensure all necessary features exist (optional, e.g., for scaler alignment)
            for feat in expected_features:
                if feat not in df_mapped.columns:
                    df_mapped[feat] = 0
            st.info("✅ UNSW-NB15 features aligned for model training.")
        elif dataset_type == "NSL":
            df_mapped = df_orig.copy()
            for feat in expected_features:
                if feat not in df_mapped.columns:
                    df_mapped[feat] = 0
            st.info("NSL-KDD features aligned (missing ones filled with 0).")
        else:
            st.warning("⚠️ Unknown dataset type — trying to align with zeros.")
            df_mapped = df_orig.copy()
            for feat in expected_features:
                if feat not in df_mapped.columns:
                    df_mapped[feat] = 0

        st.subheader("🛠 Mapped Data Preview")
        st.dataframe(df_mapped.head(10), use_container_width=True)

        # -----------------------
        # 5) Prediction options
        # -----------------------
        st.subheader("🎯 Prediction Options")
        col1, col2 = st.columns(2)
        with col1:
            prediction_type = st.selectbox(
                "Prediction Type",
                ["All Records", "Sample (First 10)", "Custom Range"]
            )
        with col2:
            start_idx, end_idx = 0, 0
            if prediction_type == "Custom Range":
                start_idx = st.number_input("Start Index", min_value=0, max_value=len(df_mapped)-1, value=0)
                end_idx = st.number_input("End Index", min_value=start_idx, max_value=len(df_mapped)-1,
                                        value=min(start_idx+9, len(df_mapped)-1))

        # -----------------------
        # 6) Run predictions
        # -----------------------
        if st.button("🚀 Process Predictions", type="primary", use_container_width=True):
            if prediction_type == "All Records":
                subset = df_mapped.copy()
                orig_subset = df_orig.copy()
            elif prediction_type == "Sample (First 10)":
                subset = df_mapped.head(10).copy()
                orig_subset = df_orig.head(10).copy()
            else:
                subset = df_mapped.iloc[start_idx:end_idx+1].copy()
                orig_subset = df_orig.iloc[start_idx:end_idx+1].copy()

            # Scale features
            encoders = models["encoders"]

            try:
                X_scaled, _ = preprocess_input(
                    subset,
                    encoders= encoders,
                    label_col=label_col
                )
            except Exception as e:
                st.error(f"Preprocessing error: {e}")
                return

            # Ground truth
            y_true_batch = None
            if label_col:
                raw_labels = orig_subset[label_col].values
                try:
                    y_true_batch = np.array(raw_labels).astype(int)
                    if not set(np.unique(y_true_batch)).issubset({0, 1}):
                        raise ValueError
                except Exception:
                    y_true_batch = np.array([0 if str(x).lower() == "normal" else 1 for x in raw_labels])

            # Predict
            results_df, metrics = predict_attack(models, X_scaled, y_true_batch)

            # Show results
            st.subheader("📈 Prediction Results")
            st.dataframe(results_df, use_container_width=True)

            
            if y_true_batch is not None and metrics is not None:
                st.subheader("✅ Evaluation Metrics")
                col1, col2, col3, col4 = st.columns(4)
                with col1: st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                with col2: st.metric("Precision", f"{metrics['precision']:.4f}")
                with col3: st.metric("Recall", f"{metrics['recall']:.4f}")
                with col4: st.metric("F1 Score", f"{metrics['f1']:.4f}")

                # 📊 Plot Metrics as Bar Chart
                metrics_df = pd.DataFrame({
                    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
                    "Value": [
                        metrics['accuracy'],
                        metrics['precision'],
                        metrics['recall'],
                        metrics['f1']
                    ]
                })

                # Download button
                csv = results_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Results",
                    data=csv,
                    file_name=f"prediction_results_{uploaded_file.name}",
                    mime="text/csv"
                )


def model_info_page(model):
    """Display model information"""
    st.header("🤖 Model Information")
    
    st.subheader("Ensemble Model")
    
    # Display model type
    st.write(f"**Model**: {type(model).__name__}")
    st.write("**File**: ong.pkl")
    st.write("**Type**: Ensemble Model (combines multiple algorithms)")
    
    st.subheader("Feature Information")
    st.write("The model uses the following key features:")
    
    features = get_feature_names()
    for i, feature in enumerate(features, 1):
        st.write(f"{i}. {feature}")
    
    st.subheader("Complete Dataset Structure")
    st.write("The NSL-KDD dataset contains the following columns:")
    
    complete_features = get_complete_column_names()
    for i, feature in enumerate(complete_features, 1):
        st.write(f"{i}. {feature}")
    
    st.subheader("Model Performance")
    st.info("""
    The model was trained on the NSL-KDD dataset and achieves high accuracy 
    in detecting various types of network intrusions including:
    - DoS (Denial of Service)
    - Probe attacks
    - R2L (Remote to Local)
    - U2R (User to Root)
    """)

def about_page():
    """About page"""
    st.header("ℹ️ About This System")
    
    st.markdown("""
    ### Intrusion Detection System
    
    This web application provides real-time network traffic analysis to detect potential security intrusions.
    
    #### Features:
    - **Real-time Analysis**: Analyze network traffic patterns instantly
    - **Multiple Models**: Uses ensemble of machine learning models for better accuracy
    - **User-friendly Interface**: Simple form-based input for network features
    - **Detailed Results**: Provides confidence scores and feature analysis
    
    #### How it works:
    1. Enter network traffic features in the form
    2. Click "Analyze Traffic" to process the data
    3. View the prediction results and confidence scores
    4. Examine feature importance for insights
    
    #### Dataset:
    - Trained on NSL-KDD dataset
    - Contains various types of network attacks
    - Includes both normal and malicious traffic patterns
    
    #### Technology Stack:
    - **Backend**: Python, Scikit-learn, TensorFlow
    - **Frontend**: Streamlit
    - **Model**: Ensemble Model (ong.pkl)
    """)

if __name__ == "__main__":
    main()
