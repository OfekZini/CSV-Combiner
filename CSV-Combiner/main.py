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
    if not lines: return ""
    if len(lines) > 0:
        processed_lines.append(lines[0])
        data_lines = lines[1:]
    else:
        data_lines = []
    for line in data_lines:
        if not line.strip(): continue
        matches = list(re.finditer(new_doc_entry_marker_pattern, line))
        if len(matches) > 0:
            split_points = [0] + [m.start() for m in matches]
            current_segment_start = 0
            for sp in split_points[1:]:
                segment = line[current_segment_start:sp].strip()
                if segment: processed_lines.append(segment)
                current_segment_start = sp
            final_segment = line[current_segment_start:].strip()
            if final_segment: processed_lines.append(final_segment)
        else:
            processed_lines.append(line)
    return "\n".join(processed_lines)


def parse_header(header_text: str) -> List[str]:
    if not header_text.strip(): return []
    if ',' in header_text:
        try:
            return next(csv.reader([header_text]))
        except:
            return [col.strip().strip('"') for col in header_text.split(',')]
    else:
        return [col.strip().strip('"') for col in header_text.split('\n') if col.strip()]


def validate_csv_structure(csv_content: str, expected_header: List[str], existing_doc_names: set) -> Tuple[
    List[List[str]], List[Dict]]:
    valid_rows, faulty_rows = [], []
    processed_csv_content = preprocess_csv_content(csv_content)
    try:
        rows = list(csv.reader(io.StringIO(processed_csv_content)))
    except csv.Error as e:
        faulty_rows.append({"line_number": 0, "content": processed_csv_content, "error": f"CSV Parsing Error: {e}",
                            "id": "parse_error"})
        return valid_rows, faulty_rows
    if not rows: return valid_rows, faulty_rows
    actual_header = [col.strip() for col in rows[0]]
    if actual_header != expected_header:
        faulty_rows.append(
            {"line_number": 1, "content": ",".join(rows[0]), "error": f"Header mismatch", "id": "header_mismatch"})
        return valid_rows, faulty_rows
    expected_col_count = len(expected_header)
    current_batch_doc_names = set()
    for i, row in enumerate(rows[1:], start=2):
        row_content = ",".join(row)
        if not row_content.strip(): continue
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
            faulty_rows.append(
                {"line_number": i, "content": row_content, "error": row_error, "id": f"faulty_{i}_{row_content}"})
        else:
            valid_rows.append(row)
            if doc_name: existing_doc_names.add(doc_name)
    return valid_rows, faulty_rows


def create_download_csv(header: List[str], valid_rows: List[List[str]]) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(header)
    writer.writerows(valid_rows)
    return output.getvalue()


def main():
    st.set_page_config(page_title="CSV Validator Pro", layout="wide")
    st.title("📊 CSV Validator Pro")
    st.markdown("A better way to validate, fix, and process your CSV data.")

    # --- Initialize Session State ---
    if 'expected_header' not in st.session_state: st.session_state['expected_header'] = []
    if 'existing_doc_names' not in st.session_state: st.session_state['existing_doc_names'] = set()
    if 'combined_valid_rows' not in st.session_state: st.session_state['combined_valid_rows'] = []
    if 'combined_faulty_rows' not in st.session_state: st.session_state['combined_faulty_rows'] = []
    if 'ignored_rows' not in st.session_state: st.session_state['ignored_rows'] = []

    # --- Callback functions to handle data processing ---
    def process_data_content(csv_content):
        """Generic function to process any CSV content."""
        if not csv_content:
            return

        with st.spinner("Validating..."):
            valid, faulty = validate_csv_structure(
                csv_content,
                st.session_state.expected_header,
                st.session_state.existing_doc_names
            )
            # Use sets to avoid adding exact duplicate rows
            existing_valid_rows_tuples = {tuple(r) for r in st.session_state.combined_valid_rows}
            for r in valid:
                if tuple(r) not in existing_valid_rows_tuples:
                    st.session_state.combined_valid_rows.append(r)

            existing_faulty_ids = {r['id'] for r in st.session_state.combined_faulty_rows}
            for r in faulty:
                if r['id'] not in existing_faulty_ids:
                    st.session_state.combined_faulty_rows.append(r)

            st.toast(f"Processing complete. Found {len(valid)} new valid and {len(faulty)} new faulty rows.", icon="🎉")

    def process_pasted_text_callback():
        """Callback for the 'Process Pasted CSV' button."""
        csv_content = st.session_state.get("csv_text_input", "")
        process_data_content(csv_content)
        st.session_state.csv_text_input = ""  # Clear the input field

    def process_uploaded_file_callback():
        """Callback for when a file is uploaded."""
        uploaded_file = st.session_state.get("file_uploader")
        if uploaded_file:
            csv_content = uploaded_file.read().decode('utf-8')
            process_data_content(csv_content)

    # --- Header Input ---
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
                st.session_state.ignored_rows = []
                st.rerun()
            else:
                st.error("Header cannot be empty.")

    if not st.session_state.expected_header:
        st.warning("Please set the expected header to begin validation.")
        st.stop()

    st.success(f"**Expected Header:** `{'`, `'.join(st.session_state.expected_header)}`")

    # --- CSV Data Input with Callbacks ---

    st.header("2. Add CSV Data")
    col1, col2 = st.columns(2)

    with col1:
        with st.container():
            st.markdown(
                """
                <div style="border: 2px solid #ccc; padding: 10px; border-radius: 10px;">
                """,
                unsafe_allow_html=True
            )

            st.file_uploader(
                "Upload a CSV file (processes automatically)",
                type=['csv'],
                key="file_uploader",
                on_change=process_uploaded_file_callback
            )

            st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        with st.container():
            st.markdown(
                """
                <div style="border: 2px solid #ccc; padding: 10px; border-radius: 10px;">
                """,
                unsafe_allow_html=True
            )

            st.text_area("Or paste CSV text", height=160, key="csv_text_input")
            st.button(
                "Process Pasted CSV",
                on_click=process_pasted_text_callback,
                type="primary"
            )

            st.markdown("</div>", unsafe_allow_html=True)

    # --- Display Metrics ---
    st.header("3. Review Results")
    v_col, f_col, i_col = st.columns(3)
    v_col.metric("✅ Total Valid Rows", len(st.session_state.combined_valid_rows))
    f_col.metric("❌ Total Faulty Rows", len(st.session_state.combined_faulty_rows))
    i_col.metric("🙈 Total Ignored Rows", len(st.session_state.ignored_rows))

    # --- Faulty Row Handling ---
    if st.session_state.combined_faulty_rows:
        st.subheader("Action Required: Fix or Ignore Faulty Rows")
        st.markdown(
            "Edit rows directly in the table, then click **Re-Validate**. Or, click **Ignore** to remove them from this list.")

        # Callback for Re-Validate button
        def revalidate_row_callback(faulty_row_id, edited_row_data):
            # Find the actual faulty row by its ID
            current_faulty_row = next(
                (row for row in st.session_state.combined_faulty_rows if row['id'] == faulty_row_id), None)

            if not current_faulty_row:
                st.warning("Error: Original faulty row not found. Page may have reloaded.", icon="⚠️")
                st.rerun()  # Rerun to refresh the state
                return

            if not edited_row_data.empty:
                edited_row_list = edited_row_data.iloc[0].tolist()
                edited_row_str = ",".join(map(str, edited_row_list))
                mini_csv = "\n".join([','.join(st.session_state.expected_header), edited_row_str])

                # Temporarily remove the original doc_name associated with this faulty row
                # from existing_doc_names for re-validation
                original_doc_name = current_faulty_row['content'].split(',')[0].strip().strip('"')
                if original_doc_name in st.session_state.existing_doc_names:
                    # Only remove if it was actually part of existing_doc_names from a previously validated row
                    # This prevents removing doc names that might be valid in other existing rows.
                    st.session_state.existing_doc_names.discard(
                        original_doc_name)  # Use discard as it doesn't raise error if not found

                valid, new_faulty = validate_csv_structure(
                    mini_csv,
                    st.session_state.expected_header,
                    st.session_state.existing_doc_names  # Pass updated existing_doc_names
                )

                if valid:
                    st.session_state.combined_valid_rows.extend(valid)
                    # Remove the fixed row from faulty rows using a list comprehension
                    st.session_state.combined_faulty_rows = [
                        row for row in st.session_state.combined_faulty_rows if row['id'] != faulty_row_id
                    ]
                    st.toast("Row fixed and moved to valid data!", icon="✅")
                    st.rerun()
                else:
                    # Update the error message for the current faulty row
                    current_faulty_row_index = next(
                        (idx for idx, row in enumerate(st.session_state.combined_faulty_rows) if
                         row['id'] == faulty_row_id), None)
                    if current_faulty_row_index is not None and new_faulty:
                        st.session_state.combined_faulty_rows[current_faulty_row_index]['error'] = new_faulty[0][
                            'error']
                    st.warning(f"Still faulty: {new_faulty[0]['error']}", icon="❌")
                    # Re-add the original doc_name if it was removed and the row is still faulty
                    if original_doc_name:
                        st.session_state.existing_doc_names.add(original_doc_name)

        # Callback for Ignore button
        def ignore_row_callback(faulty_row_id):
            ignored_row_data = None
            new_faulty_rows = []
            for row in st.session_state.combined_faulty_rows:
                if row['id'] == faulty_row_id:
                    ignored_row_data = row
                else:
                    new_faulty_rows.append(row)

            if ignored_row_data:
                st.session_state.ignored_rows.append(ignored_row_data)
                st.session_state.combined_faulty_rows = new_faulty_rows  # Update the list
                st.toast(f"Row ignored.", icon="🙈")
                st.rerun()  # Rerun after modification
            else:
                st.warning("Error: Row to ignore not found. Already processed?", icon="⚠️")
                st.rerun()  # Rerun to refresh state

        # Iterate directly over the session state list; callbacks will handle modification and reruns.
        # However, if a button is pressed and triggers a rerun, the loop needs to be robust.
        # The `KeyError: 0` implies that `combined_faulty_rows` might have been empty or
        # the index was off after a rerun caused by a prior button press.
        # We need to ensure that the list isn't being iterated while it's being modified
        # by a callback on the *same* execution.
        # Streamlit's rerunning mechanism makes this tricky.
        # The best approach is often to have a flag or a temporary list to manage what needs to be removed/added
        # and then apply these changes after the loop, as was previously attempted,
        # but with proper re-initialization of the action list.
        # Let's revert slightly but apply the `id` concept.

        # New approach: Use a list to mark IDs for removal/processing and apply changes at the end of the loop.
        # This prevents "KeyError" from pop() on an already changed list.
        rows_to_remove_ids = set()
        rows_to_add_to_valid = []

        # We display each faulty row. Buttons for each row will trigger callbacks.
        # The callbacks will update session state directly and cause a rerun.
        # The problem is when one button is clicked, `st.rerun()` happens, the script restarts,
        # and the loop tries to re-render potentially removed elements.

        # Let's simplify the loop to just display, and let the callbacks handle the state changes completely.
        # The `on_click` callbacks will manage `st.session_state` and `st.rerun()`.
        # The `for` loop itself should not *assume* `combined_faulty_rows` remains static.

        # A simple iteration is fine if callbacks trigger immediate reruns.
        # The `KeyError: 0` on `original_doc_name` suggests the `faulty_row` itself
        # passed to the callback might be stale, or the index is wrong.
        # The `id` passing should fix this.

        # Iterate over a copy to ensure stable iteration during button clicks
        # when `st.rerun` might be triggered.
        # The crucial part is that the `faulty_row` in the loop's current iteration
        # should be the *exact* one we are trying to act on.
        # This is where passing `faulty_row['id']` to the callback is powerful.

        for faulty_row_item in list(st.session_state.combined_faulty_rows):  # Iterate over a copy
            st.error(f"**Error in line ~{faulty_row_item['line_number']}**: {faulty_row_item['error']}", icon="❗️")
            try:
                parsed_content = next(csv.reader(io.StringIO(faulty_row_item['content'])))
            except (StopIteration, csv.Error):
                parsed_content = [faulty_row_item['content']]
            num_cols = len(st.session_state.expected_header)
            row_data = (parsed_content + [''] * num_cols)[:num_cols]
            df_edit = pd.DataFrame([row_data], columns=st.session_state.expected_header)

            # Key the data_editor with the faulty row's ID for persistence
            # Use `f"editor_{faulty_row_item['id']}"` for unique keys
            edited_df = st.data_editor(df_edit, key=f"editor_{faulty_row_item['id']}", num_rows="dynamic")

            btn_col1, btn_col2, _ = st.columns([1, 1, 4])

            # Pass the unique ID of the faulty row to the callback
            btn_col1.button(
                "🔧 Re-Validate",
                key=f"fix_{faulty_row_item['id']}",
                type="primary",
                on_click=revalidate_row_callback,
                args=(faulty_row_item['id'], edited_df)
            )
            btn_col2.button(
                "🙈 Ignore",
                key=f"ignore_{faulty_row_item['id']}",
                on_click=ignore_row_callback,
                args=(faulty_row_item['id'],)
            )
            st.divider()

    # --- Final Data Display & Download ---
    st.header("4. Final Data")
    if st.session_state.combined_valid_rows:
        with st.expander("✅ View Valid Data", expanded=True):
            valid_df = pd.DataFrame(st.session_state.combined_valid_rows, columns=st.session_state.expected_header)
            st.dataframe(valid_df, use_container_width=True)
            st.download_button(label="⬇️ Download Clean CSV", data=create_download_csv(st.session_state.expected_header,
                                                                                       st.session_state.combined_valid_rows),
                               file_name="clean_data.csv", mime="text/csv", type="primary")
    if st.session_state.ignored_rows:
        with st.expander("🙈 View Ignored Rows"):
            ignored_data_for_display = [
                {"Original Line": r['line_number'], "Error": r['error'], "Original Content": r['content']} for r in
                st.session_state.ignored_rows]
            st.table(ignored_data_for_display)


if __name__ == "__main__":
    main()