CREATE TABLE documents (
  id TEXT PRIMARY KEY,                       -- ULID
  title TEXT NOT NULL,
  source_sha256 TEXT NOT NULL UNIQUE,
  source_path TEXT NOT NULL,                 -- relative to data_dir, always a PDF
  original_name TEXT NOT NULL,
  page_count INTEGER NOT NULL,
  status TEXT NOT NULL,                      -- see section 5.1
  priority INTEGER NOT NULL DEFAULT 0,
  ingest_requested INTEGER NOT NULL DEFAULT 0,
  ingest_status TEXT NOT NULL DEFAULT 'none',
  delivered_version INTEGER,                 -- last final build attached to the chat, section 9.7
  export_name TEXT UNIQUE,                   -- folder name in export_dir, section 9.7
  exported_version INTEGER,                  -- last final build copied to export_dir
  export_error TEXT,                         -- last export failure, retried by the worker
  export_attempt_at TEXT,                    -- last export attempt, retries at most every 10 minutes
  rebuild_requested INTEGER NOT NULL DEFAULT 0,   -- set by rebuild, cleared by a successful final build, section 5.1
  redecide_requested INTEGER NOT NULL DEFAULT 0,  -- set by ocr redecide, cleared when no page needs re-deciding, section 6.11
  error TEXT,
  created_at TEXT NOT NULL,                  -- ISO 8601 UTC
  updated_at TEXT NOT NULL
);

CREATE TABLE profiles (
  id TEXT PRIMARY KEY,                       -- OCR version: sha256 of canonical JSON, section 6.11
  name TEXT NOT NULL,                        -- pass1 | pass2 | region | pdf_text
  config_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE pages (
  id INTEGER PRIMARY KEY,
  document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  page_index INTEGER NOT NULL,               -- 0-based position in source
  printed_label TEXT,
  text_layer TEXT,                           -- none | hidden | visible; null until checked
  track TEXT,                                -- vlm | pdf_text; null until checked, section 6.1
  status TEXT NOT NULL DEFAULT 'pending',    -- pending | ok | flagged | failed | reviewed
  current_attempt_id INTEGER REFERENCES attempts(id),    -- null for a failed page without a candidate
  current_revision_id INTEGER REFERENCES revisions(id),
  flags_json TEXT NOT NULL DEFAULT '[]',     -- hard and signal flags, section 6.9
  signals_json TEXT,                         -- signal counters and pass comparison, section 6.10
  review_json TEXT,                          -- review item with hints and item_version, section 6.13
  rules_version TEXT,                        -- rules version the page was decided by, section 6.11
  ocr_generation INTEGER NOT NULL DEFAULT 1,  -- incremented by reprocess and retry, section 6.11
  retry_at TEXT,                             -- next attempt after a transient model error, section 12.3
  retry_count INTEGER NOT NULL DEFAULT 0,
  error TEXT,                                -- last permanent error of a failed page
  line_boxes_json TEXT,
  line_boxes_version TEXT,                   -- LINES_IMPL_VERSION + detector parameters, section 6.11
  updated_at TEXT NOT NULL,
  UNIQUE (document_id, page_index)
);
CREATE INDEX pages_doc_status ON pages(document_id, status);

CREATE TABLE attempts (
  id INTEGER PRIMARY KEY,
  page_id INTEGER NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
  profile_id TEXT NOT NULL REFERENCES profiles(id),
  generation INTEGER NOT NULL,               -- pages.ocr_generation at request start; reused only when equal
  pass_no INTEGER NOT NULL,                  -- 1 or 2; 0 for pdf_text
  role TEXT NOT NULL,                        -- full | part | combined | pdf_text
  region_json TEXT,                          -- part: {part, of, rect}; combined: {parts: [ids]}
  text TEXT NOT NULL,
  finish_reason TEXT,
  tokens_in INTEGER,
  tokens_out INTEGER,
  duration_ms INTEGER,
  token_probs_json TEXT,                     -- raw token probabilities (counts + tokens with p < 0.99), section 6.6; null without logprobs
  created_at TEXT NOT NULL
);
CREATE INDEX attempts_page ON attempts(page_id);

CREATE TABLE revisions (
  id INTEGER PRIMARY KEY,
  page_id INTEGER NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
  request_id TEXT NOT NULL UNIQUE,           -- idempotency key from the review window, section 8.4
  base_attempt_id INTEGER REFERENCES attempts(id),
  base_revision_id INTEGER REFERENCES revisions(id),
  action TEXT NOT NULL,                      -- accept (text equals the base) | edit
  scope TEXT NOT NULL,                       -- hints | full, section 6.13
  text TEXT NOT NULL,
  printed_label TEXT,
  choices_json TEXT,                         -- A/B choices made in the window, for the pilot report
  duration_s INTEGER,                        -- time from opening the page to "Done"
  created_at TEXT NOT NULL
);
CREATE INDEX revisions_page ON revisions(page_id);

CREATE TABLE drafts (
  page_id INTEGER PRIMARY KEY REFERENCES pages(id) ON DELETE CASCADE,
  item_version TEXT NOT NULL,
  base_attempt_id INTEGER REFERENCES attempts(id),
  base_revision_id INTEGER REFERENCES revisions(id),
  text TEXT NOT NULL,
  printed_label TEXT,
  choices_json TEXT,
  hints_json TEXT,                           -- hints shifted to the edited text, with their states, section 8.4
  cursor INTEGER,                            -- UTF-16 code units, section 11
  scope TEXT,                                -- scope the review was started with, section 8.1
  seq INTEGER NOT NULL DEFAULT 1,            -- save number; PUT is a conditional update on it, section 8.4
  stale INTEGER NOT NULL DEFAULT 0,          -- saved with keep_stale: the window asks "keep mine / take new", section 8.4
  created_at TEXT NOT NULL,                  -- age for the stale-draft reminder, section 8.1
  updated_at TEXT NOT NULL
);

CREATE TABLE samples (                       -- spot checks, section 15.4
  id INTEGER PRIMARY KEY,
  page_id INTEGER NOT NULL UNIQUE REFERENCES pages(id) ON DELETE CASCADE,
  created_at TEXT NOT NULL,
  done_at TEXT
);

CREATE TABLE builds (
  id INTEGER PRIMARY KEY,
  document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  version INTEGER NOT NULL,
  kind TEXT NOT NULL DEFAULT 'final',        -- final | draft, section 9
  status TEXT NOT NULL,                      -- pending (draft only) | running | done | failed
  output_dir TEXT,
  manifest_json TEXT,
  error TEXT,
  pruned_at TEXT,
  created_at TEXT NOT NULL,
  UNIQUE (document_id, version)
);

CREATE TABLE ingests (
  id INTEGER PRIMARY KEY,
  build_id INTEGER NOT NULL UNIQUE REFERENCES builds(id) ON DELETE CASCADE,   -- one ingest per final build
  status TEXT NOT NULL,                      -- pending | done | failed; a retry moves the same row back to pending
  response_json TEXT,
  error TEXT,
  created_at TEXT NOT NULL
);

CREATE TABLE gold (
  page_id INTEGER PRIMARY KEY REFERENCES pages(id) ON DELETE CASCADE,
  text TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE control (key TEXT PRIMARY KEY, value TEXT NOT NULL);   -- worker_heartbeat
