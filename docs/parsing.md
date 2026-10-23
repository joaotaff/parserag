# Parsing

Ingest turns each PDF into LlamaIndex nodes before anything is embedded. The work is in `src/parserag/parse.py`, called from `ingest_dataset` in `src/parserag/index.py`.

```
PDF -> LlamaParse JSON -> data/parsed/{dataset}/[{tier}/]{doc_id}/pages.json
     -> drop blank pages -> SentenceSplitter -> nodes with dataset, doc_id, page
```

A later ingest of a doc already listed in the selected tier's `indexed.json` skips this path. `--rebuild` clears that tier's manifest and reparses from its cache, which still avoids a new LlamaParse job.

## LlamaParse

`load_pages` returns the cache when `pages.json` exists. Otherwise `_llama_parse()` builds a `LlamaParse` client and calls `get_json_result`. The client needs `LLAMA_CLOUD_API_KEY`.

`_llama_parse()` passes `api_key`, `result_type="markdown"`, `ignore_errors=False`, `verbose=False`, and the CLI-selected tier. Named tiers also send `version=PARSE_VERSION` (`"latest"`); the API rejects a tier without a version. `--parse-tier` accepts `fast`, `cost_effective`, `agentic`, and `agentic_plus`; omitting it or passing `none` leaves both `tier` and `version` unset for the API's accurate default. Every other field below is the default from `llama-cloud-services` 0.6.94. `base_url` uses `LLAMA_CLOUD_BASE_URL` when that variable is set. Runs record `parse_tier` and `parse_version` on `run_config`.

### Client

| Setting | Value |
| --- | --- |
| `api_key` | `LLAMA_CLOUD_API_KEY` via `llama_cloud_api_key()` |
| `base_url` | `https://api.cloud.llamaindex.ai` |
| `organization_id` | `None` |
| `project_id` | `None` |
| `check_interval` | `1` |
| `backoff_pattern` | `"linear"` |
| `max_check_interval` | `5` |
| `custom_client` | `None` |
| `ignore_errors` | `False` |
| `max_timeout` | `2000` |
| `num_workers` | `4` |
| `result_type` | `"markdown"` |
| `show_progress` | `True` |
| `split_by_page` | `True` |
| `verbose` | `False` |
| `is_remote` | `False` |
| `check_for_updates` | `False` |

### Parse options

| Setting | Value |
| --- | --- |
| `adaptive_long_table` | `False` |
| `aggressive_table_extraction` | `False` |
| `annotate_links` | `False` |
| `auto_mode` | `False` |
| `auto_mode_configuration_json` | `None` |
| `auto_mode_trigger_on_image_in_page` | `False` |
| `auto_mode_trigger_on_table_in_page` | `False` |
| `auto_mode_trigger_on_text_in_page` | `None` |
| `auto_mode_trigger_on_regexp_in_page` | `None` |
| `azure_openai_api_version` | `None` |
| `azure_openai_deployment_name` | `None` |
| `azure_openai_endpoint` | `None` |
| `azure_openai_key` | `None` |
| `bbox_bottom` | `None` |
| `bbox_left` | `None` |
| `bbox_right` | `None` |
| `bbox_top` | `None` |
| `compact_markdown_table` | `False` |
| `continuous_mode` | `False` |
| `disable_ocr` | `False` |
| `disable_image_extraction` | `False` |
| `do_not_cache` | `False` |
| `do_not_unroll_columns` | `False` |
| `extract_charts` | `False` |
| `extract_layout` | `False` |
| `fast_mode` | `False` |
| `guess_xlsx_sheet_name` | `False` |
| `high_res_ocr` | `False` |
| `html_make_all_elements_visible` | `False` |
| `html_remove_fixed_elements` | `False` |
| `html_remove_navigation_elements` | `False` |
| `http_proxy` | `None` |
| `ignore_document_elements_for_layout_detection` | `False` |
| `inline_images_in_markdown` | `False` |
| `input_s3_region` | `None` |
| `invalidate_cache` | `False` |
| `job_timeout_extra_time_per_page_in_seconds` | `None` |
| `job_timeout_in_seconds` | `None` |
| `keep_page_separator_when_merging_tables` | `False` |
| `language` | `"en"` |
| `markdown_table_multiline_header_separator` | `None` |
| `max_pages` | `None` |
| `merge_tables_across_pages_in_markdown` | `False` |
| `output_pdf_of_document` | `False` |
| `output_s3_path_prefix` | `None` |
| `output_s3_region` | `None` |
| `output_tables_as_HTML` | `False` |
| `outlined_table_extraction` | `False` |
| `page_error_tolerance` | `None` |
| `page_prefix` | `None` |
| `page_separator` | `None` |
| `page_suffix` | `None` |
| `parse_mode` | `None` |
| `premium_mode` | `False` |
| `preset` | `None` |
| `preserve_layout_alignment_across_pages` | `False` |
| `preserve_very_small_text` | `False` |
| `presentation_out_of_bounds_content` | `False` |
| `precise_bounding_box` | `False` |
| `replace_failed_page_mode` | `None` |
| `replace_failed_page_with_error_message_prefix` | `None` |
| `replace_failed_page_with_error_message_suffix` | `None` |
| `remove_hidden_text` | `False` |
| `save_images` | `True` |
| `skip_diagonal_text` | `False` |
| `spreadsheet_extract_sub_tables` | `False` |
| `spreadsheet_force_formula_computation` | `False` |
| `specialized_chart_parsing_agentic` | `False` |
| `specialized_chart_parsing_efficient` | `False` |
| `specialized_chart_parsing_plus` | `False` |
| `specialized_image_parsing` | `False` |
| `strict_mode_buggy_font` | `False` |
| `strict_mode_image_extraction` | `False` |
| `strict_mode_image_ocr` | `False` |
| `strict_mode_reconstruction` | `False` |
| `structured_output` | `False` |
| `structured_output_json_schema` | `None` |
| `structured_output_json_schema_name` | `None` |
| `system_prompt` | `None` |
| `system_prompt_append` | `None` |
| `take_screenshot` | `False` |
| `target_pages` | `None` |
| `user_prompt` | `None` |
| `vendor_multimodal_api_key` | `None` |
| `vendor_multimodal_model_name` | `None` |
| `model` | `None` |
| `webhook_url` | `None` |
| `partition_pages` | `None` |
| `hide_headers` | `False` |
| `hide_footers` | `False` |
| `page_header_suffix` | `None` |
| `page_header_prefix` | `None` |
| `page_footer_suffix` | `None` |
| `page_footer_prefix` | `None` |
| `extract_printed_page_number` | `None` |
| `line_level_bounding_box` | `False` |
| `tier` | CLI-selected; `None` by default (`PARSE_TIER`) |
| `version` | `"latest"` when a named tier is set; else `None` |

### Deprecated

These fields are still on the client and still at their SDK defaults.

| Setting | Value |
| --- | --- |
| `bounding_box` | `None` |
| `complemental_formatting_instruction` | `None` |
| `content_guideline_instruction` | `None` |
| `formatting_instruction` | `None` |
| `gpt4o_mode` | `False` |
| `gpt4o_api_key` | `None` |
| `is_formatting_instruction` | `False` |
| `parsing_instruction` | `""` |
| `use_vendor_multimodal_model` | `False` |

JSON is used because each page object carries its own page number. The download also includes layout data, so a long 10-K can sit on that call for several minutes with no further log lines.

The result may be a list, a dict, or a pydantic object. The parser reads `pages` from the first job, then for each page:

- Page number: `page`, else `page_number`, else the 1-based position in the result.
- Text: `md`, else `markdown`, else `text`. Missing text becomes `""`.

If the smallest page number in the file is `0`, every page is shifted by +1. Internal pages are 1-indexed, matching gold pages (see `docs/datasets.md`). A result that is already 1-indexed is stored unchanged.

`pages.json` is written only after a result that contains at least one page. The file is a JSON list of `{"page": <int>, "text": "<markdown>"}`. A raised error is not cached. A result with zero pages raises and is not cached either.

## Nodes

`parse_to_nodes` loads those pages and drops any whose text is empty or whitespace. Each remaining page becomes one `Document` with metadata `dataset`, `doc_id`, and `page`.

`SentenceSplitter(chunk_size=1024, chunk_overlap=128)` then splits any page over 1024 tokens. Shorter pages stay one node. Every chunk keeps the parent page's metadata, so retrieval eval can still match a gold `(doc_id, page)` after a split. `dataset` is listed in `excluded_embed_metadata_keys`, so it is stored on the node and left out of the embedding text.

If every page was blank, the node list is empty. That `pages.json` stays on disk. Ingest records the doc in `failures.json` and does not index it. Delete that doc's `pages.json` to force a new LlamaParse call.

## Failures during ingest

`ingest_dataset` parses one PDF at a time. On an exception it logs the doc, appends `{doc_id, error}` to the selected tier's `failures.json`, and continues. The command exits 1 while that file exists. The next ingest for that tier retries docs that are not in its `indexed.json`.

A successful index of a previously failed doc removes it from `failures.json`. When the list is empty, the file is deleted.
