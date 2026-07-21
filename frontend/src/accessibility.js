function nonblank(value) {
  return typeof value === "string" && value.trim() ? value.trim() : "";
}

export function promptSegmentAnnouncement(segment) {
  if (!segment || typeof segment !== "object") return nonblank(segment);
  const kind = segment.kind ?? segment.type ?? "text";
  if (kind === "math") {
    return (
      nonblank(segment.spoken_text) ||
      nonblank(segment.expression) ||
      nonblank(segment.latex) ||
      nonblank(segment.math)
    );
  }
  if (kind === "blank") {
    return nonblank(segment.label) || "Answer blank";
  }
  if (kind === "table" || kind === "plot") {
    return (
      nonblank(segment.spoken_text) ||
      nonblank(segment.description) ||
      nonblank(segment.caption) ||
      nonblank(segment.title)
    );
  }
  return nonblank(segment.text);
}

export function promptSegmentsAnnouncement(segments) {
  if (!Array.isArray(segments)) return "";
  return segments
    .map(promptSegmentAnnouncement)
    .filter(Boolean)
    .join(" ");
}

function contentBlockAnnouncement(block) {
  if (!block || typeof block !== "object") return "";
  const label =
    block.kind === "worked_example"
      ? "Worked example."
      : block.kind === "remediation"
        ? "Review."
        : "";
  return [
    label,
    nonblank(block.text),
    promptSegmentsAnnouncement(block.segments),
  ]
    .filter(Boolean)
    .join(" ");
}

export function transcriptEntryAnnouncement(entry) {
  if (!entry || typeof entry !== "object") return "";
  const structured = Array.isArray(entry.content_blocks)
    ? entry.content_blocks.map(contentBlockAnnouncement).filter(Boolean).join(" ")
    : "";
  const prompt = promptSegmentsAnnouncement(entry.prompt_segments);
  const content = structured || prompt || nonblank(entry.text);
  if (!content) return "";
  const speaker = entry.role === "student" ? "Your response" : "Tutor update";
  return `${speaker}: ${content}`;
}
