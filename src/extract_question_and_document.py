import re
from pathlib import Path

import openpyxl
import pandas as pd


def _parse_documents(cell_value: str) -> list[str]:
    """Parse a cell like '1. doc1 \n2. doc2' into ['doc1', 'doc2']."""
    parts = re.split(r"\d+\.\s*", cell_value)
    return [p.strip() for p in parts if p.strip()]


def extract_questions_and_documents(
    file_path: Path,
    sheet_name: str,
    start_col: int,
    document_row: int,
    question_row: int,
) -> pd.DataFrame:
    """Read an Excel sheet and return a DataFrame pairing each question with its documents.

    Args:
        file_path: Path to the .xlsx file.
        sheet_name: Name of the sheet to read.
        start_col: First column (1-indexed) containing data.
        document_row: Row number (1-indexed) containing documents.
        question_row: Row number (1-indexed) containing questions.

    Returns:
        DataFrame with columns: question, question_ref, document_ref, 1, 2, ...
    """
    wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
    ws = wb[sheet_name]

    rows: list[dict] = []
    max_docs = 0

    for col_idx in range(start_col, ws.max_column + 1):
        question_cell = ws.cell(row=question_row, column=col_idx)
        document_cell = ws.cell(row=document_row, column=col_idx)

        question = question_cell.value
        document_raw = document_cell.value

        if question is None or document_raw is None:
            continue

        question = str(question).strip()
        document_raw = str(document_raw).strip()

        if not question or not document_raw:
            continue

        docs = _parse_documents(document_raw)
        if not docs:
            continue

        col_letter = openpyxl.utils.get_column_letter(col_idx)
        row_data: dict = {
            "question": question,
            "question_ref": f"{col_letter}{question_row}",
            "document_ref": f"{col_letter}{document_row}",
        }
        for i, doc in enumerate(docs, start=1):
            row_data[str(i)] = doc

        max_docs = max(max_docs, len(docs))
        rows.append(row_data)

    wb.close()

    if not rows:
        raise ValueError("No valid question/document pairs found in the sheet.")

    doc_columns = [str(i) for i in range(1, max_docs + 1)]
    df = pd.DataFrame(rows, columns=["question", "question_ref", "document_ref"] + doc_columns)
    return df
