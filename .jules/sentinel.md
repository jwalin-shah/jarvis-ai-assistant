## 2026-02-18 - CSV Injection Vulnerability
**Vulnerability:** CSV Injection (Formula Injection) in export functionality.
**Learning:** User input from iMessage (sender names, message text) can contain spreadsheet formulas starting with =, +, -, or @. When exported to CSV and opened in Excel, these can execute arbitrary commands.
**Prevention:** Sanitize all user-controlled fields in CSV exports by prepending a single quote (') if they start with dangerous characters. Added `_sanitize_csv_field` helper in `jarvis/export.py`.
