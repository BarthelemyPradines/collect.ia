from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.extract_question_and_document import extract_questions_and_documents
from src.llm.ask_with_documents import ask_question_with_documents
from src.llm.structured_output import parse_structured_output_spec

load_dotenv()


def main():
    df = extract_questions_and_documents(
        file_path=Path("data/ANALYSE DOE  SITE TUNNEL trame.xlsx"),
        sheet_name="TUNNELS",
        start_col=3,
        document_row=1,
        question_row=3,
        category_row=2,
        structured_output_row=4,
    )
    print(df.to_string(index=False))
    print()

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    df.to_csv(output_dir / "questions.csv", index=False)

    # Test LLM call with the first row
    row = df.iloc[1]
    doc_columns = [c for c in df.columns if c.isdigit()]
    document_names = [row[c] for c in doc_columns if isinstance(row[c], str)]

    print(f"Question: {row['question']}")
    print(f"Documents: {document_names}")

    spec = row.get("structured_output")
    response_model = parse_structured_output_spec(spec) if pd.notna(spec) else None
    if response_model:
        print(f"Structured output: {spec}")
    print(response_model)
    
    answer = ask_question_with_documents(
        question=row["question"],
        document_names=document_names,
        documents_dir=Path("data"),
        use_remote=True,
        response_model=response_model,
    )
    print(f"Answer: {answer}")


if __name__ == "__main__":
    main()
