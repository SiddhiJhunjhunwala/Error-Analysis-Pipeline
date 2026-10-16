"""Google Sheets access: sign-in, retries, chunked writes, formatting, and reading/writing by column name."""
import random
import time
from collections import Counter

import gspread
import numpy as np
import pandas as pd

from .config import resolve_path
from .text import col_letter

NEW_TABS = ["pairs", "manual_annotation", "llm_labels", "agreement"]
CELL_LIMIT = 10_000_000  # Google Sheets limit per spreadsheet


def connect(cfg):
    """Open the spreadsheet with OAuth (your Google account) or a service account."""
    a = cfg["google_auth"]
    if a["method"] == "service_account":
        path = resolve_path(cfg, a["service_account_file"])
        if not path.exists():
            raise FileNotFoundError(f"Service account file not found: {path}")
        gc = gspread.service_account(filename=str(path))
    else:
        client = resolve_path(cfg, a["oauth_client_file"])
        if not client.exists():
            raise FileNotFoundError(f"OAuth client file not found: {client}. See 'Google sign-in' in README.md.")
        gc = gspread.oauth(credentials_filename=str(client),
                           authorized_user_filename=str(resolve_path(cfg, a["token_file"])))
    sh = gc.open_by_url(cfg["sheet"]["url"])
    print(f"Connected to: {sh.title}")
    return sh


def with_retry(fn, *args, **kwargs):
    """Retry on rate limits (429) and server errors with exponential backoff."""
    for attempt in range(7):
        try:
            return fn(*args, **kwargs)
        except gspread.exceptions.APIError as e:
            code = getattr(getattr(e, "response", None), "status_code", None)
            if code in (429, 500, 502, 503) and attempt < 6:
                wait = min(64, 2 ** attempt) + random.random()
                print(f"  Sheets API {code}; retrying in {wait:.0f}s")
                time.sleep(wait)
            else:
                raise


def to_cell(v):
    if v is None:
        return ""
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return "" if np.isnan(v) else float(v)
    return v


def write_rows(ws, rows, start_row=1, start_col=1, option="RAW", max_chars=1_500_000):
    """Write a 2D list in chunks small enough for the API. RAW keeps text starting with '=' as text."""
    chunk, size, r0 = [], 0, start_row
    for row in rows:
        s = sum(len(str(v)) for v in row) + 8 * len(row)
        if chunk and size + s > max_chars:
            with_retry(ws.update, range_name=f"{col_letter(start_col)}{r0}", values=chunk, value_input_option=option)
            r0, chunk, size = r0 + len(chunk), [], 0
        chunk.append(row)
        size += s
    if chunk:
        with_retry(ws.update, range_name=f"{col_letter(start_col)}{r0}", values=chunk, value_input_option=option)


rgb = lambda h: {"red": int(h[0:2], 16) / 255, "green": int(h[2:4], 16) / 255, "blue": int(h[4:6], 16) / 255}
DARK, GOLD, LIGHT_YELLOW, WHITE = "404040", "BF9000", "FFF2CC", "FFFFFF"


def grid(sid, r0, r1, c0, c1):
    return {"sheetId": sid, "startRowIndex": r0, "endRowIndex": r1, "startColumnIndex": c0, "endColumnIndex": c1}


def fmt(rng, **f):
    return {"repeatCell": {"range": rng, "cell": {"userEnteredFormat": f},
                           "fields": ",".join(f"userEnteredFormat.{k}" for k in f)}}


def header_fmt(rng, color=DARK):
    return fmt(rng, backgroundColor=rgb(color), wrapStrategy="WRAP", verticalAlignment="TOP",
               textFormat={"bold": True, "foregroundColor": rgb(WHITE)})


def col_width(sid, c, chars, hidden=False):
    props = {"pixelSize": int(chars * 7)}
    fields = "pixelSize"
    if hidden:
        props["hiddenByUser"], fields = True, "pixelSize,hiddenByUser"
    return {"updateDimensionProperties": {"range": {"sheetId": sid, "dimension": "COLUMNS",
                                                    "startIndex": c, "endIndex": c + 1},
                                          "properties": props, "fields": fields}}


def table_requests(ws, df, widths=None, wrap=(), inputs=(), hidden=(), notes=None):
    sid, n, cols = ws.id, len(df), list(df.columns)
    req = [header_fmt(grid(sid, 0, 1, 0, len(cols))),
           {"updateSheetProperties": {"properties": {"sheetId": sid, "gridProperties":
                                                     {"frozenRowCount": 1, "frozenColumnCount": 1}},
                                      "fields": "gridProperties.frozenRowCount,gridProperties.frozenColumnCount"}},
           {"setBasicFilter": {"filter": {"range": grid(sid, 0, n + 1, 0, len(cols))}}}]
    for j, c in enumerate(cols):
        req.append(col_width(sid, j, (widths or {}).get(c, 14), hidden=c in hidden))
        if c in inputs:
            req.append(header_fmt(grid(sid, 0, 1, j, j + 1), GOLD))
            req.append(fmt(grid(sid, 1, n + 1, j, j + 1), backgroundColor=rgb(LIGHT_YELLOW)))
        if c in wrap:
            req.append(fmt(grid(sid, 1, n + 1, j, j + 1), wrapStrategy="WRAP", verticalAlignment="TOP"))
        if notes and c in notes:
            req.append({"updateCells": {"rows": [{"values": [{"note": notes[c]}]}], "fields": "note",
                                        "start": {"sheetId": sid, "rowIndex": 0, "columnIndex": j}}})
    return req


def df_rows(df, header=True):
    rows = [list(df.columns)] if header else []
    rows += [[to_cell(v) for v in row] for row in df.itertuples(index=False)]
    return rows


# ---------- reading and writing tabs by column name ----------
def read_tab_df(sh, tab):
    """A whole tab as a DataFrame of strings; blank headers become _col<N>."""
    vals = with_retry(sh.worksheet(tab).get_all_values)
    if not vals:
        return pd.DataFrame()
    w = max(len(r) for r in vals)
    vals = [r + [""] * (w - len(r)) for r in vals]
    seen, header = Counter(), []
    for j, h in enumerate(vals[0]):
        h = h.strip() or f"_col{j + 1}"
        seen[h] += 1
        header.append(h if seen[h] == 1 else f"{h}_{seen[h]}")
    return pd.DataFrame(vals[1:], columns=header, dtype=object)


def write_columns(sh, tab, updates, key="pair_id"):
    """Write the columns of `updates` (index = pair_id) into `tab`, matching rows by `key`.
    Other rows keep their values; columns missing from the tab are added at the end."""
    ws = sh.worksheet(tab)
    header = with_retry(ws.row_values, 1)
    keys = with_retry(ws.col_values, header.index(key) + 1)[1:]
    pos = {k: i for i, k in enumerate(keys)}
    unknown = [k for k in updates.index if k not in pos]
    if unknown:
        print(f"  Warning: {len(unknown)} ids not found in '{tab}', skipped (e.g. {unknown[:3]})")
    new_cols = [c for c in updates.columns if c not in header]
    if new_cols:
        need = len(header) + len(new_cols)
        if ws.col_count < need:
            with_retry(ws.add_cols, need - ws.col_count)
        with_retry(ws.update, range_name=f"{col_letter(len(header) + 1)}1", values=[new_cols], value_input_option="RAW")
        with_retry(sh.batch_update, {"requests": [header_fmt(grid(ws.id, 0, 1, len(header), need))]})
        header = header + new_cols
    ranges = [f"{col_letter(header.index(c) + 1)}2:{col_letter(header.index(c) + 1)}{len(keys) + 1}"
              for c in updates.columns]
    got = with_retry(ws.batch_get, ranges)
    data = []
    for c, g, rng in zip(updates.columns, got, ranges):
        col = [g[k][0] if k < len(g) and g[k] else "" for k in range(len(keys))]
        for pid, v in updates[c].items():
            if pid in pos:
                col[pos[pid]] = to_cell(v)
        data.append({"range": rng, "values": [[v] for v in col]})
    with_retry(ws.batch_update, data, value_input_option="RAW")


def write_blocks(sh, ws, blocks, start_row, start_col, stamp=None):
    """Write titled tables stacked vertically as values, with bold titles and dark headers."""
    out, title_rows, header_rows = [], [], []
    if stamp:
        out += [[stamp], []]
    for title, t in blocks:
        title_rows.append(len(out)); out.append([title])
        header_rows.append((len(out), len(t.columns))); out.append([str(c) for c in t.columns])
        out += df_rows(t, header=False)
        out.append([])
    width = max(len(r) for r in out)
    out = [r + [""] * (width - len(r)) for r in out]
    need_rows, need_cols = start_row + len(out), start_col + width
    if ws.row_count < need_rows:
        with_retry(ws.add_rows, need_rows - ws.row_count)
    if ws.col_count < need_cols:
        with_retry(ws.add_cols, need_cols - ws.col_count)
    with_retry(ws.batch_clear, [f"{col_letter(start_col)}{start_row}:{col_letter(ws.col_count)}{ws.row_count}"])
    sid, r0, c0 = ws.id, start_row - 1, start_col - 1
    req = [{"repeatCell": {"range": grid(sid, r0, ws.row_count, c0, ws.col_count), "cell": {},
                           "fields": "userEnteredFormat"}}]
    write_rows(ws, out, start_row=start_row, start_col=start_col)
    if stamp:
        req.append(fmt(grid(sid, r0, r0 + 1, c0, c0 + 1), textFormat={"italic": True}))
    for i in title_rows:
        req.append(fmt(grid(sid, r0 + i, r0 + i + 1, c0, c0 + 1), textFormat={"bold": True}))
    for i, w in header_rows:
        req.append(header_fmt(grid(sid, r0 + i, r0 + i + 1, c0, c0 + w)))
    with_retry(sh.batch_update, {"requests": req})


