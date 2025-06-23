import streamlit as st
import pandas as pd
import io
import csv
import re
from typing import List, Dict, Tuple


def parse_header(header_text: str) -> List[str]:
    """Parse the expected header from input text"""
    if not header_text.strip():
        return []

    # Handle both comma-separated and newline-separated headers
    if ',' in header_text:
        # Use CSV reader to properly handle quoted strings with commas
        try:
            csv_reader = csv.reader([header_text])
            parsed_header = next(csv_reader)
            return [col.strip().strip('"') for col in parsed_header]
        except:
            # Fallback to simple split if CSV parsing fails
            return [col.strip().strip('"') for col in header_text.split(',')]
    else:
        return [col.strip().strip('"') for col in header_text.split('\n') if col.strip()]


def split_pdf_lines(raw_text: str) -> str:
    """
    Preprocess raw CSV text to ensure that each line contains only one PDF document.
    Splits lines if more than one `.pdf`-like document name is detected.
    """
    lines = raw_text.strip().split('\n')
    fixed_lines = []

    for line in lines:
        matches = list(re.finditer(r'[^,\s"]+\.pdf', line))  # Find all .pdf names
        if len(matches) <= 1:
            fixed_lines.append(line)
        else:
            # Split line by PDF positions
            split_positions = [m.start() for m in matches]
            segments = []
            for i, pos in enumerate(split_positions):
                segment = line[pos:] if i == len(split_positions) - 1 else line[pos:split_positions[i + 1]]
                # Prepend previous text if it's not the first chunk
                if i > 0:
                    segment = line[split_positions[i - 1]:split_positions[i]] + segment
                segments.append(segment.strip(', \n'))

            fixed_lines.extend(segments)

    return "\n".join(fixed_lines)


def validate_csv_structure(csv_content: str, expected_header: List[str]) -> Tuple[List[List[str]], List[Dict]]:
    """
    Validate CSV content against expected header
    Returns: (valid_rows, faulty_rows_with_errors)
    """
    valid_rows = []
    faulty_rows = []

    # Parse CSV content
    csv_reader = csv.reader(io.StringIO(csv_content))
    rows = list(csv_reader)

    if not rows:
        return valid_rows, [{"line_number": 0, "content": "", "error": "Empty CSV content"}]

    # Check header
    actual_header = [col.strip() for col in rows[0]]
    expected_header_clean = [col.strip() for col in expected_header]

    if actual_header != expected_header_clean:
        faulty_rows.append({
            "line_number": 1,
            "content": ",".join(rows[0]),
            "error": f"Header mismatch. Expected: {expected_header_clean}, Got: {actual_header}"
        })
        return valid_rows, faulty_rows

    # Validate data rows
    expected_col_count = len(expected_header_clean)

    for i, row in enumerate(rows[1:], start=2):  # Start from line 2 (after header)
        row_content = ",".join(row)

        # Skip rows that are just 'csv'
        if row_content.strip().lower() == 'csv':
            continue

        # Check column count
        if len(row) != expected_col_count:
            faulty_rows.append({
                "line_number": i,
                "content": row_content,
                "error": f"Column count mismatch. Expected {expected_col_count} columns, got {len(row)}"
            })
            continue

        # Check for empty values (optional - you can modify this logic)
        empty_cols = [j for j, cell in enumerate(row) if not cell.strip()]
        if empty_cols:
            col_names = [expected_header_clean[j] for j in empty_cols]
            faulty_rows.append({
                "line_number": i,
                "content": row_content,
                "error": f"Empty values in columns: {', '.join(col_names)}"
            })
            continue

        # If all checks pass, add to valid rows
        valid_rows.append(row)

    return valid_rows, faulty_rows


def create_combined_csv(header: List[str], valid_rows: List[List[str]], faulty_rows: List[Dict]) -> str:
    """Create combined CSV content with valid data and faulty rows marked"""
    output = io.StringIO()
    writer = csv.writer(output)

    # Write header with additional error column
    extended_header = header + ['ERROR_DESCRIPTION']
    writer.writerow(extended_header)

    # Write valid rows (with empty error column)
    for row in valid_rows:
        writer.writerow(row + [''])

    # Write faulty rows with error descriptions
    for faulty_row in faulty_rows:
        # Parse the original content to get individual columns
        try:
            row_data = list(csv.reader([faulty_row['content']]))[0]
            # Pad or truncate to match header length
            while len(row_data) < len(header):
                row_data.append('')
            row_data = row_data[:len(header)]
            writer.writerow(row_data + [faulty_row['error']])
        except:
            # If parsing fails, just add the raw content
            writer.writerow([''] * len(header) + [f"Line {faulty_row['line_number']}: {faulty_row['error']}"])

    return output.getvalue()


def create_download_csv(header: List[str], valid_rows: List[List[str]]) -> str:
    """Create CSV content for download"""
    output = io.StringIO()
    writer = csv.writer(output)

    # Write header
    writer.writerow(header)

    # Write valid rows
    for row in valid_rows:
        writer.writerow(row)

    return output.getvalue()


def main():
    st.set_page_config(page_title="CSV Header Validator", layout="wide")

    st.title("📊 CSV Header Validator and Processor")
    st.markdown("Upload or paste CSV data to validate against a predefined header structure.")

    # Header input on main page
    st.header("Expected CSV Header")
    st.markdown("Enter the expected column headers (comma-separated or one per line):")

    expected_header_input = st.text_area(
        "Header columns:",
        placeholder="name,email,age,city\nor\nname\nemail\nage\ncity",
        height=100,
        key="header_input"
    )

    # Button to set header
    header_submitted = st.button("Set Header", type="primary", help="Click to set the expected header structure")

    expected_header = []
    if header_submitted and expected_header_input.strip():
        expected_header = parse_header(expected_header_input)
        st.session_state['expected_header'] = expected_header

    # Use session state to persist header
    if 'expected_header' in st.session_state:
        expected_header = st.session_state['expected_header']

    if expected_header:
        st.success(f"Expected columns ({len(expected_header)}): {', '.join(expected_header)}")
    else:
        st.warning("Please enter and set the expected header structure")

    # Main content area
    if expected_header:
        st.header("CSV Data Input")

        # Show both input methods side by side
        col1, col2 = st.columns(2)

        csv_content = ""

        with col1:
            st.subheader("📝 Paste CSV Text")
            csv_text = st.text_area(
                "CSV Content:",
                placeholder="name,email,age,city\nJohn Doe,john@email.com,30,New York\nJane Smith,jane@email.com,25,Los Angeles",
                height=250,
                key="csv_text_input"
            )
            text_submitted = st.button("Process Pasted CSV", type="primary", key="process_text")

            if text_submitted and csv_text.strip():
                csv_content = csv_text
                st.session_state['csv_content'] = csv_content
                st.session_state['input_method'] = 'text'

        with col2:
            st.subheader("📁 Upload CSV File")
            uploaded_file = st.file_uploader(
                "Choose a CSV file",
                type=['csv'],
                help="Upload a CSV file to validate",
                key="file_uploader"
            )

            # Add some spacing to match the text area height
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")

            file_submitted = st.button("Process Uploaded CSV", type="primary", key="process_file")

            if file_submitted and uploaded_file is not None:
                try:
                    csv_content = uploaded_file.read().decode('utf-8')
                    st.session_state['csv_content'] = csv_content
                    st.session_state['input_method'] = 'file'
                    st.session_state['filename'] = uploaded_file.name
                    st.success(f"File '{uploaded_file.name}' processed successfully!")

                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")

        # Use session state to get CSV content
        if 'csv_content' in st.session_state:
            csv_content = st.session_state['csv_content']

        # Process CSV if content is available
        if csv_content.strip():
            st.header("🔍 Validation Results")

            with st.spinner("Validating CSV structure..."):
                cleaned_csv_content = split_pdf_lines(csv_content)
                valid_rows, faulty_rows = validate_csv_structure(cleaned_csv_content, expected_header)

                # Combine with existing data if any
                if 'combined_valid_rows' not in st.session_state:
                    st.session_state['combined_valid_rows'] = []
                if 'combined_faulty_rows' not in st.session_state:
                    st.session_state['combined_faulty_rows'] = []

                # Add new valid rows to combined data
                st.session_state['combined_valid_rows'].extend(valid_rows)

                # Add new faulty rows to combined data (with updated line numbers)
                line_offset = len(st.session_state['combined_valid_rows']) + len(
                    st.session_state['combined_faulty_rows'])
                for faulty_row in faulty_rows:
                    faulty_row_copy = faulty_row.copy()
                    faulty_row_copy['original_line_number'] = faulty_row_copy['line_number']
                    faulty_row_copy['line_number'] = line_offset + faulty_row_copy['line_number']
                    st.session_state['combined_faulty_rows'].append(faulty_row_copy)

                # Update session state with current totals
                st.session_state['valid_rows'] = st.session_state['combined_valid_rows']
                st.session_state['faulty_rows'] = st.session_state['combined_faulty_rows']

            # Get combined totals
            total_valid = len(st.session_state['combined_valid_rows'])
            total_faulty = len(st.session_state['combined_faulty_rows'])

            # Display results
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("New Valid", len(valid_rows))

            with col2:
                st.metric("New Faulty", len(faulty_rows))

            with col3:
                st.metric("Total Valid", total_valid)

            with col4:
                st.metric("Total Faulty", total_faulty)

            # Clear data button
            if st.button("🗑️ Clear All Data", help="Clear all combined data and start fresh"):
                st.session_state['combined_valid_rows'] = []
                st.session_state['combined_faulty_rows'] = []
                st.session_state['valid_rows'] = []
                st.session_state['faulty_rows'] = []
                st.rerun()

            # Show faulty rows with editing capability
            if total_faulty > 0:
                st.subheader("❌ Faulty Rows - Edit and Fix")
                st.markdown("Edit the faulty rows below and click 'Fix Selected Rows' to validate them:")

                # Create editable form for faulty rows
                with st.form("edit_faulty_rows"):
                    edited_rows = []
                    rows_to_fix = []

                    for i, faulty_row in enumerate(st.session_state['combined_faulty_rows']):
                        col1, col2, col3 = st.columns([1, 3, 1])

                        with col1:
                            fix_this_row = st.checkbox(f"Fix", key=f"fix_{i}",
                                                       help=f"Check to fix line {faulty_row['line_number']}")

                        with col2:
                            edited_content = st.text_input(
                                f"Line {faulty_row['line_number']}:",
                                value=faulty_row['content'],
                                key=f"edit_{i}",
                                help=f"Error: {faulty_row['error']}"
                            )
                            st.caption(f"❌ {faulty_row['error']}")

                        with col3:
                            st.metric("Line", faulty_row['line_number'])

                        if fix_this_row:
                            edited_rows.append({
                                'index': i,
                                'content': edited_content,
                                'original': faulty_row
                            })
                            rows_to_fix.append(i)

                    # Submit button for fixing rows
                    fix_submitted = st.form_submit_button("🔧 Fix Selected Rows", type="primary")

                    if fix_submitted and edited_rows:
                        # Validate edited rows
                        newly_valid = []
                        still_faulty = []

                        for edited_row in edited_rows:
                            # Create a mini CSV to validate
                            mini_csv = ",".join(expected_header) + "\n" + edited_row['content']
                            valid, faulty = validate_csv_structure(mini_csv, expected_header)

                            if valid and len(valid) > 0:
                                newly_valid.extend(valid)
                                st.success(f"✅ Fixed line {edited_row['original']['line_number']}")
                            else:
                                # Keep as faulty with updated content
                                updated_faulty = edited_row['original'].copy()
                                updated_faulty['content'] = edited_row['content']
                                if faulty and len(faulty) > 0:
                                    updated_faulty['error'] = faulty[0]['error']
                                still_faulty.append(updated_faulty)
                                st.error(f"❌ Line {edited_row['original']['line_number']} still has issues")

                        # Update session state
                        # Remove fixed rows from faulty list
                        remaining_faulty = [row for i, row in enumerate(st.session_state['combined_faulty_rows']) if
                                            i not in rows_to_fix]
                        # Add still faulty rows back
                        remaining_faulty.extend(still_faulty)

                        # Update session state
                        st.session_state['combined_faulty_rows'] = remaining_faulty
                        st.session_state['combined_valid_rows'].extend(newly_valid)
                        st.session_state['valid_rows'] = st.session_state['combined_valid_rows']
                        st.session_state['faulty_rows'] = st.session_state['combined_faulty_rows']

                        if newly_valid:
                            st.success(f"🎉 Successfully fixed {len(newly_valid)} rows!")

                        st.rerun()

                # Show remaining faulty rows in a table
                if st.session_state['combined_faulty_rows']:
                    st.subheader("📋 Remaining Faulty Rows Summary")
                    faulty_df = pd.DataFrame(st.session_state['combined_faulty_rows'])
                    st.dataframe(
                        faulty_df[['line_number', 'content', 'error']],
                        use_container_width=True,
                        column_config={
                            "line_number": "Line #",
                            "content": "Row Content",
                            "error": "Error Description"
                        }
                    )
            else:
                st.success("✅ All rows are valid!")

            # Show valid data preview
            if total_valid > 0:
                st.subheader("✅ Combined Valid Data Preview")

                # Create DataFrame for valid data
                valid_df = pd.DataFrame(st.session_state['combined_valid_rows'], columns=expected_header)
                st.dataframe(valid_df, use_container_width=True)

                # Download buttons for clean CSV
                col1, col2, col3 = st.columns(3)

                with col1:
                    clean_csv = create_download_csv(expected_header, st.session_state['combined_valid_rows'])
                    st.download_button(
                        label="📥 Download Clean CSV",
                        data=clean_csv,
                        file_name="validated_data.csv",
                        mime="text/csv",
                        help="Download the validated CSV with only valid rows"
                    )

                with col2:
                    # Combined CSV with all data (valid + faulty marked)
                    combined_csv = create_combined_csv(expected_header, st.session_state['combined_valid_rows'],
                                                       st.session_state['combined_faulty_rows'])
                    st.download_button(
                        label="📥 Download Combined CSV",
                        data=combined_csv,
                        file_name="combined_data.csv",
                        mime="text/csv",
                        help="Download CSV with valid data and faulty rows marked with error descriptions"
                    )

                with col3:
                    # Faulty rows report
                    if st.session_state['combined_faulty_rows']:
                        faulty_df = pd.DataFrame(st.session_state['combined_faulty_rows'])
                        faulty_csv = faulty_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Faulty Report",
                            data=faulty_csv,
                            file_name="faulty_rows_report.csv",
                            mime="text/csv",
                            help="Download a report of all faulty rows with error descriptions"
                        )

                # Show statistics
                with st.expander("📊 Data Statistics"):
                    total_rows = total_valid + total_faulty
                    st.write(f"**Total rows processed:** {total_rows}")
                    st.write(f"**Valid rows:** {total_valid}")
                    st.write(f"**Faulty rows:** {total_faulty}")
                    if total_rows > 0:
                        success_rate = (total_valid / total_rows) * 100
                        st.write(f"**Success rate:** {success_rate:.1f}%")

    else:
        st.info("Please define the expected CSV header structure above to begin validation.")

    # Always show current combined CSV at the bottom if data exists
    if 'combined_valid_rows' in st.session_state and 'expected_header' in st.session_state:
        st.markdown("---")
        st.header("📋 Current Combined CSV Preview")

        valid_rows = st.session_state.get('combined_valid_rows', [])
        faulty_rows = st.session_state.get('combined_faulty_rows', [])
        expected_header = st.session_state.get('expected_header', [])

        if valid_rows or faulty_rows:
            # Create combined preview
            all_data = []
            status_column = []

            # Add valid rows
            for row in valid_rows:
                all_data.append(row)
                status_column.append('✅ Valid')

            # Add faulty rows
            for faulty_row in faulty_rows:
                try:
                    row_data = list(csv.reader([faulty_row['content']]))[0]
                    while len(row_data) < len(expected_header):
                        row_data.append('')
                    row_data = row_data[:len(expected_header)]
                    all_data.append(row_data)
                    status_column.append(f"❌ {faulty_row['error']}")
                except:
                    all_data.append([''] * len(expected_header))
                    status_column.append(f"❌ Line {faulty_row['line_number']}: {faulty_row['error']}")

            # Create DataFrame safely
            try:
                if all_data:
                    # Ensure all rows have the same number of columns
                    max_cols = len(expected_header)
                    normalized_data = []
                    for row in all_data:
                        if len(row) < max_cols:
                            row.extend([''] * (max_cols - len(row)))
                        elif len(row) > max_cols:
                            row = row[:max_cols]
                        normalized_data.append(row)

                    # Create DataFrame with normalized data
                    combined_df = pd.DataFrame(normalized_data, columns=expected_header)
                    combined_df['Status'] = status_column

                    st.dataframe(combined_df, use_container_width=True)

                    # Quick download button
                    combined_csv = create_combined_csv(expected_header, valid_rows, faulty_rows)
                    st.download_button(
                        label="📥 Quick Download Combined CSV",
                        data=combined_csv,
                        file_name="combined_data.csv",
                        mime="text/csv",
                        help="Download CSV with all data including error descriptions"
                    )
                else:
                    st.info("No data to display")
            except Exception as e:
                st.error(f"Error displaying data: {str(e)}")
                st.write("Raw data for debugging:")
                st.write(f"Expected header length: {len(expected_header)}")
                st.write(f"Valid rows: {len(valid_rows)}")
                st.write(f"Faulty rows: {len(faulty_rows)}")
                if all_data:
                    st.write(f"Sample row lengths: {[len(row) for row in all_data[:3]]}")

    # Footer
    st.markdown("---")
    st.markdown(
        "**Note:** This tool validates CSV structure, column count, and checks for empty values. You can modify the validation logic as needed.")


if __name__ == "__main__":
    main()