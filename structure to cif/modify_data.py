# fe_excel_processor.py
import ast
import pandas as pd
from typing import Any, Dict, Optional


class FormationEnergyExcelProcessor:
    """
    A class to process Excel files containing 'formula' and 'formation_energy' columns.
    
    Workflow:
    - Read the input Excel sheet
    - Safely parse the 'formation_energy' field (string → dict)
    - Build a new DataFrame with:
        * file_name       → formatted as "<row_number+2>_<formula>.cif"
        * value_per_atom  → extracted from formation_energy dict
        * value           → extracted from formation_energy dict
    - Save the result to a new Excel file with a specified sheet name
    """

    def __init__(
        self,
        input_file: str = "1_MatDX_EF.xlsx",
        input_sheet: str = "1_MatDX_EF",
        output_file: str = "1_MatDX_EF_modified.xlsx",
        output_sheet: str = "Sheet1",
    ):
        self.input_file = input_file
        self.input_sheet = input_sheet
        self.output_file = output_file
        self.output_sheet = output_sheet

    # ---------- Helpers ----------
    @staticmethod
    def _to_dict_safe(x: Any) -> Dict[str, Any]:
        """Safely convert a string/object into a dictionary, or return empty dict if parsing fails."""
        if isinstance(x, dict):
            return x
        if pd.isna(x):
            return {}
        try:
            return ast.literal_eval(str(x))
        except Exception:
            return {}

    @staticmethod
    def _get_val(d: Dict[str, Any], key: str) -> Optional[float]:
        """Extract a value from a dictionary safely, return None if key is missing."""
        try:
            return d.get(key, None)
        except Exception:
            return None

    # ---------- Main workflow ----------
    def load(self) -> pd.DataFrame:
        """Load the input Excel sheet into a DataFrame and validate required columns."""
        df = pd.read_excel(self.input_file, sheet_name=self.input_sheet)
        if "formation_energy" not in df.columns or "formula" not in df.columns:
            raise KeyError("Required columns (formula, formation_energy) are missing.")
        return df

    def build_processed_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the original DataFrame into a new one containing file_name, value_per_atom, and value.
        """
        fe_series = df["formation_energy"].apply(self._to_dict_safe)
        new_df = pd.DataFrame({
            "file_name": [f"{i+2}_{str(f)}.cif" for i, f in enumerate(df["formula"])],
            "value_per_atom": [self._get_val(d, "value_per_atom") for d in fe_series],
            "value": [self._get_val(d, "value") for d in fe_series],
        })
        return new_df

    def save(self, processed: pd.DataFrame) -> None:
        """Save the processed DataFrame to a new Excel file with the given sheet name."""
        with pd.ExcelWriter(self.output_file, engine="openpyxl", mode="w") as writer:
            processed.to_excel(writer, sheet_name=self.output_sheet, index=False)

    def run(self) -> str:
        """
        Execute the full pipeline: load → process → save.
        Returns the path of the output file.
        """
        df = self.load()
        processed = self.build_processed_df(df)
        self.save(processed)
        print(f"Done: '{self.output_file}' has been created with only the '{self.output_sheet}' sheet.")
        return self.output_file


if __name__ == "__main__":
    processor = FormationEnergyExcelProcessor(
        input_file="1_MatDX_EF.xlsx",
        input_sheet="1_MatDX_EF",
        output_file="1_MatDX_EF_modified.xlsx",
        output_sheet="Sheet1",
    )
    processor.run()
