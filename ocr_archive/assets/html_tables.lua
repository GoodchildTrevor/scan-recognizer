-- Turn raw HTML blocks (HTML tables written by the model) into pandoc tables.
-- The Markdown builder passes every HTML table as one fenced raw block (```{=html}),
-- so the whole table reaches this filter at once.
function RawBlock(el)
  if el.format == "html" then
    return pandoc.read(el.text, "html").blocks
  end
end
