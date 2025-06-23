import streamlit as st
import pandas as pd
import io
import csv
from typing import List, Dict, Tuple


def parse_header(header_text: str) -> List[str]:
    """Parse the expected header from input text"""
    if not header_text.strip():
        return []

    # Handle both comma-separated and newline-separated headers
    if ',' in header_text:
        return [col.strip() for col in header_text.split(',')]
    else:
        return [col.strip() for col in header_text.split('\n') if col.strip()]


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

    # Sidebar for expected header input
    with st.sidebar:
        st.header("Expected CSV Header")
        st.markdown("Enter the expected column headers (comma-separated or one per line):")

        expected_header_input = st.text_area(
            "Header columns:",
            placeholder="name,email,age,city\nor\nname\nemail\nage\ncity",
            height=150
        )

        expected_header = parse_header(expected_header_input)

        if expected_header:
            st.success(f"Expected columns ({len(expected_header)}):")
            for i, col in enumerate(expected_header, 1):
                st.write(f"{i}. {col}")
        else:
            st.warning("Please enter the expected header structure")

    # Main content area
    if expected_header:
        st.header("CSV Data Input")

        # Create tabs for different input methods
        tab1, tab2 = st.tabs(["📝 Paste CSV Text", "📁 Upload CSV File"])

        csv_content = ""

        with tab1:
            st.markdown("Paste your CSV content below:")
            csv_text = st.text_area(
                "CSV Content:",
                placeholder="name,email,age,city\nJohn Doe,john@email.com,30,New York\nJane Smith,jane@email.com,25,Los Angeles",
                height=200
            )
            if csv_text.strip():
                csv_content = csv_text

        with tab2:
            uploaded_file = st.file_uploader(
                "Choose a CSV file",
                type=['csv'],
                help="Upload a CSV file to validate"
            )

            if uploaded_file is not None:
                try:
                    csv_content = uploaded_file.read().decode('utf-8')
                    st.success(f"File '{uploaded_file.name}' uploaded successfully!")

                    # Show preview
                    with st.expander("Preview uploaded content"):
                        st.text(csv_content[:500] + "..." if len(csv_content) > 500 else csv_content)

                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")

        # Process CSV if content is available
        if csv_content.strip():
            st.header("🔍 Validation Results")

            with st.spinner("Validating CSV structure..."):
                valid_rows, faulty_rows = validate_csv_structure(csv_content, expected_header)

            # Display results
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Valid Rows", len(valid_rows))

            with col2:
                st.metric("Faulty Rows", len(faulty_rows))

            # Show faulty rows if any
            if faulty_rows:
                st.subheader("❌ Faulty Rows")
                st.markdown("The following rows have issues:")

                faulty_df = pd.DataFrame(faulty_rows)
                st.dataframe(
                    faulty_df,
                    use_container_width=True,
                    column_config={
                        "line_number": "Line #",
                        "content": "Row Content",
                        "error": "Error Description"
                    }
                )

                # Option to download faulty rows report
                faulty_csv = faulty_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Faulty Rows Report",
                    data=faulty_csv,
                    file_name="faulty_rows_report.csv",
                    mime="text/csv",
                    help="Download a report of all faulty rows with error descriptions"
                )
            else:
                st.success("✅ All rows are valid!")

            # Show valid data preview
            if valid_rows:
                st.subheader("✅ Valid Data Preview")

                # Create DataFrame for valid data
                valid_df = pd.DataFrame(valid_rows, columns=expected_header)
                st.dataframe(valid_df, use_container_width=True)

                # Download button for clean CSV
                clean_csv = create_download_csv(expected_header, valid_rows)
                st.download_button(
                    label="📥 Download Clean CSV",
                    data=clean_csv,
                    file_name="validated_data.csv",
                    mime="text/csv",
                    help="Download the validated CSV with only valid rows"
                )

                # Show statistics
                with st.expander("📊 Data Statistics"):
                    st.write(f"**Total rows processed:** {len(valid_rows) + len(faulty_rows)}")
                    st.write(f"**Valid rows:** {len(valid_rows)}")
                    st.write(f"**Faulty rows:** {len(faulty_rows)}")
                    if valid_rows:
                        success_rate = (len(valid_rows) / (len(valid_rows) + len(faulty_rows))) * 100
                        st.write(f"**Success rate:** {success_rate:.1f}%")

    else:
        st.info("👈 Please define the expected CSV header structure in the sidebar to begin validation.")

    # Footer
    st.markdown("---")
    st.markdown(
        "**Note:** This tool validates CSV structure, column count, and checks for empty values. You can modify the validation logic as needed.")


if __name__ == "__main__":
    main()