import sys; 
from src.extract_question_and_document import extract_questions_and_documents
from pathlib import Path


def main():
    print("Testing Question and Document Extraction")
    df = extract_questions_and_documents(
          file_path=Path('data/fake_data/test_excel.xlsx'),
          sheet_name='Feuil1',
          start_col=3,
          document_row=3,
          question_row=4,
      )
    print(df.to_string(index=False))
    print()
    print('Columns:', list(df.columns))
    print('Shape:', df.shape)



if __name__ == "__main__":
    main()
