import sys; 
from src.extract_question_and_document import extract_questions_and_documents
from pathlib import Path


def main():
    print("Testing Question and Document Extraction")
    df = extract_questions_and_documents(
          file_path=Path('data/ANALYSE DOE  SITE TUNNEL trame.xlsx'),
          sheet_name='TUNNELS',
          start_col=3,
          document_row=1,
          question_row=3,
          category_row=2,
      )
    print(df.to_string(index=False))
    print()
    print('Columns:', list(df.columns))
    print('Shape:', df.shape)
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    df.to_csv('outputs/ANALYSE DOE  SITE TUNNEL trame.csv', index=False)



if __name__ == "__main__":
    main()
