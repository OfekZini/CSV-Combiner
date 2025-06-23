import streamlit as st
import pandas as pd
import io
import csv
from typing import List, Dict, Tuple
import re

# --- UNCHANGED HELPER FUNCTIONS ---
def preprocess_csv_content(csv_content: str) -> str:
    lines = csv_content.splitlines()
    processed_lines = []
    new_doc_entry_marker_pattern = r'(?<!^)([0-9a-zA-Z._-]+?\.pdf)"?,\d+,'
    if not lines:
        return ""
    if len(lines) > 0:
        processed_lines.append(lines[0])
        data_lines = lines[1:]
    else:
        data_lines = []
    for line in data_lines:
        if not line.strip():
            continue
        matches = list(re.finditer(new_doc_entry_marker_pattern, line))
        if len(matches) > 0:
            split_points = [0] + [m.start() for m in matches]
            current_segment_start = 0
            for sp in split_points[1:]:
                segment = line[current_segment_start:sp].strip()
                if segment:
                    processed_lines.append(segment)
                current_segment_start = sp
            final_segment = line[current_segment_start:].strip()
            if final_segment:
                processed_lines.append(final_segment)
        else:
            processed_lines.append(line)
    return "\n".join(processed_lines)


def parse_header(header_text: str) -> List[str]:
    if not header_text.strip():
        return []
    if ',' in header_text:
        try:
            return next(csv.reader([header_text]))
        except:
            return [col.strip().strip('"') for col in header_text.split(',')]
    else:
        return [col.strip().strip('"') for col in header_text.split('\n') if col.strip()]


def validate_csv_structure(csv_content: str, expected_header: List[str], existing_doc_names: set) -> Tuple[List[List[str]], List[Dict]]:
    valid_rows, faulty_rows = [], []
    processed_csv_content = preprocess_csv_content(csv_content)
    try:
        rows = list(csv.reader(io.StringIO(processed_csv_content)))
    except csv.Error as e:
        faulty_rows.append({"line_number": 0, "content": processed_csv_content, "error": f"CSV Parsing Error: {e}", "id": "parse_error"})
        return valid_rows, faulty_rows
    if not rows:
        return valid_rows, faulty_rows
    actual_header = [col.strip() for col in rows[0]]
    if actual_header != expected_header:
        faulty_rows.append({"line_number": 1, "content": ",".join(rows[0]), "error": f"Header mismatch", "id": "header_mismatch"})
        return valid_rows, faulty_rows
    expected_col_count = len(expected_header)
    current_batch_doc_names = set()
    for i, row in enumerate(rows[1:], start=2):
        row_content = ",".join(row)
        if not row_content.strip():
            continue
        row_error, doc_name = None, ""
        if len(row) != expected_col_count:
            row_error = f"Column count mismatch. Expected {expected_col_count}, got {len(row)}."
        else:
            empty_cols = [expected_header[j] for j, cell in enumerate(row) if not cell.strip()]
            if empty_cols:
                row_error = f"Empty values in columns: {', '.join(empty_cols)}."
            else:
                doc_name = row[0].strip().strip('"')
                if doc_name in existing_doc_names or doc_name in current_batch_doc_names:
                    row_error = f"Duplicate document name found: '{doc_name}'."
                else:
                    current_batch_doc_names.add(doc_name)
        if row_error:
            faulty_rows.append({"line_number": i, "content": row_content, "error": row_error, "id": f"faulty_{i}_{row_content}"})
        else:
            valid_rows.append(row)
            if doc_name:
                existing_doc_names.add(doc_name)
    return valid_rows, faulty_rows


def create_download_csv(header: List[str], valid_rows: List[List[str]]) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(header)
    writer.writerows(valid_rows)
    return output.getvalue()


def main():
    st.set_page_config(page_title="CSV Validator Pro", layout="wide")
    st.title("\U0001F4CA CSV Validator Pro")
    st.markdown("A better way to validate and process your CSV data.")

    if 'expected_header' not in st.session_state:
        st.session_state['expected_header'] = []
    if 'existing_doc_names' not in st.session_state:
        st.session_state['existing_doc_names'] = set()
    if 'combined_valid_rows' not in st.session_state:
        st.session_state['combined_valid_rows'] = []
    if 'combined_faulty_rows' not in st.session_state:
        st.session_state['combined_faulty_rows'] = []

    def process_data_content(csv_content):
        if not csv_content:
            return
        with st.spinner("Validating..."):
            valid, faulty = validate_csv_structure(
                csv_content,
                st.session_state.expected_header,
                st.session_state.existing_doc_names
            )
            existing_valid_rows_tuples = {tuple(r) for r in st.session_state.combined_valid_rows}
            for r in valid:
                if tuple(r) not in existing_valid_rows_tuples:
                    st.session_state.combined_valid_rows.append(r)
            existing_faulty_ids = {r['id'] for r in st.session_state.combined_faulty_rows}
            for r in faulty:
                if r['id'] not in existing_faulty_ids:
                    st.session_state.combined_faulty_rows.append(r)
            st.toast(f"Processing complete. Found {len(valid)} new valid and {len(faulty)} new faulty rows.", icon="\U0001F389")

    def process_pasted_text_callback():
        csv_content = st.session_state.get("csv_text_input", "")
        process_data_content(csv_content)
        st.session_state.csv_text_input = ""

    def process_uploaded_file_callback():
        uploaded_file = st.session_state.get("file_uploader")
        if uploaded_file:
            csv_content = uploaded_file.read().decode('utf-8')
            process_data_content(csv_content)

    with st.expander("1. Define Expected CSV Header", expanded=not st.session_state.expected_header):
        expected_header_input = st.text_area(
            "Header columns (comma-separated or one per line): doc_name,number,description,country,research_focus",
            height=100, key="header_input"
        )
        if st.button("Set Header", type="primary"):
            header = parse_header(expected_header_input)
            if header:
                st.session_state.expected_header = header
                st.session_state.existing_doc_names = set()
                st.session_state.combined_valid_rows = []
                st.session_state.combined_faulty_rows = []
                st.rerun()
            else:
                st.error("Header cannot be empty.")

    if not st.session_state.expected_header:
        st.warning("Please set the expected header to begin validation.")
        st.stop()

    st.success(f"**Expected Header:** `{'`, `'.join(st.session_state.expected_header)}`")

    st.header("2. Add CSV Data")
    col1, col2 = st.columns(2)

    with col1:
        with st.container():
            st.markdown("""
                <div style="border: 2px solid #ccc; padding: 10px; border-radius: 10px;">
                """, unsafe_allow_html=True)
            st.file_uploader(
                "Upload a CSV file (processes automatically)",
                type=['csv'],
                key="file_uploader",
                on_change=process_uploaded_file_callback
            )
            st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        with st.container():
            st.markdown("""
                <div style="border: 2px solid #ccc; padding: 10px; border-radius: 10px;">
                """, unsafe_allow_html=True)
            st.text_area("Or paste CSV text", height=160, key="csv_text_input")
            st.button("Process Pasted CSV", on_click=process_pasted_text_callback, type="primary")
            st.markdown("</div>", unsafe_allow_html=True)

    st.header("3. Review Results")
    v_col, f_col = st.columns(2)
    v_col.metric("✅ Total Valid Rows", len(st.session_state.combined_valid_rows))
    f_col.metric("❌ Total Faulty Rows", len(st.session_state.combined_faulty_rows))

    st.header("4. Final Data")
    if st.session_state.combined_valid_rows:
        with st.expander("✅ View Valid Data", expanded=True):
            valid_df = pd.DataFrame(st.session_state.combined_valid_rows, columns=st.session_state.expected_header)
            st.dataframe(valid_df, use_container_width=True)
            st.download_button(
                label="⬇️ Download Clean CSV",
                data=create_download_csv(st.session_state.expected_header, st.session_state.combined_valid_rows),
                file_name="clean_data.csv",
                mime="text/csv",
                type="primary"
            )

    if st.session_state.combined_faulty_rows:
        with st.expander("❌ View Faulty Rows", expanded=False):
            faulty_data_for_display = [
                {
                    "Line Number": r['line_number'],
                    "Error": r['error'],
                    "Row Content": r['content']
                } for r in st.session_state.combined_faulty_rows
            ]
            st.dataframe(pd.DataFrame(faulty_data_for_display), use_container_width=True)


if __name__ == "__main__":
    main()
