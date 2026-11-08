// Build an IEEE-style two-column Word copy of the CCECE paper from blocks.json.
// Usage: node build_docx.js blocks.json out.docx
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun, AlignmentType,
  WidthType, BorderStyle, SectionType, TableLayoutType, VerticalAlign,
} = require("docx");

const [, , inFile, outFile] = process.argv;
const { blocks } = JSON.parse(fs.readFileSync(inFile, "utf8"));
const dir = path.dirname(inFile);
const FONT = "Times New Roman";
const IN = 1440;
const PAGE_W = 8.5 * IN, MARGIN_LR = 0.625 * IN, GAP = 0.25 * IN;
const FULL = PAGE_W - 2 * MARGIN_LR;           // 10440
const COL = (FULL - GAP) / 2;                    // 5040
const ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII"];

const run = (r, size, extra = {}) => new TextRun({
  text: r.t, font: FONT, size, bold: r.b || extra.bold, italics: r.i || extra.italics,
  superScript: r.sup, smallCaps: extra.smallCaps,
});
const runs = (rs, size, extra) => rs.map((r) => run(r, size, extra));
const pngSize = (f) => { const b = fs.readFileSync(f); return [b.readUInt32BE(16), b.readUInt32BE(20)]; };

const noBorder = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const thin = { style: BorderStyle.SINGLE, size: 4, color: "000000" };

function tableBlock(b) {
  const total = b.wide ? FULL : COL;
  const ncol = b.rows[0].length;
  // column widths from the longest text in each column
  const len = Array(ncol).fill(0);
  b.rows.forEach((row) => row.forEach((c, j) => {
    const L = c.map((r) => r.t).join("").length; len[j] = Math.max(len[j], Math.min(L, 40) + 2);
  }));
  const sum = len.reduce((a, x) => a + x, 0);
  const widths = len.map((x) => Math.floor((x / sum) * total));
  widths[ncol - 1] += total - widths.reduce((a, x) => a + x, 0);
  const align = b.align.replace(/[^lcr]/g, "");
  const size = b.wide ? 14 : 14;
  const rows = b.rows.map((row, i) => new TableRow({
    tableHeader: i === 0,
    children: row.map((c, j) => new TableCell({
      width: { size: widths[j], type: WidthType.DXA },
      verticalAlign: VerticalAlign.CENTER,
      margins: { top: 10, bottom: 10, left: 50, right: 50 },
      borders: {
        top: i === 0 ? thin : noBorder,
        bottom: i === 0 || i === b.rows.length - 1 ? thin : noBorder,
        left: noBorder, right: noBorder,
      },
      children: [new Paragraph({
        alignment: align[j] === "r" ? AlignmentType.RIGHT : align[j] === "c" ? AlignmentType.CENTER : AlignmentType.LEFT,
        spacing: { before: 0, after: 0 },
        children: runs(c, size, i === 0 ? { bold: true } : {}),
      })],
    })),
  }));
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 160, after: 0 }, keepNext: true,
      children: [new TextRun({ text: `TABLE ${b.num}`, font: FONT, size: 16 })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 0, after: 80 }, keepNext: true,
      children: runs(b.cap, 16, { smallCaps: true }) }),
    new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths, layout: TableLayoutType.FIXED, rows }),
    new Paragraph({ spacing: { before: 0, after: 120 }, children: [] }),
  ];
}

function figureBlock(b) {
  const f = path.join(dir, b.file);
  const [w, h] = pngSize(f);
  const widthPx = (b.width * COL / IN) * 96;
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 }, keepNext: true,
      children: [new ImageRun({ type: "png", data: fs.readFileSync(f),
        transformation: { width: Math.round(widthPx), height: Math.round(widthPx * h / w) } })] }),
    new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { before: 0, after: 160 },
      children: [new TextRun({ text: `Fig. ${b.num}. `, font: FONT, size: 16 }), ...runs(b.cap, 16)] }),
  ];
}

// ---- title section (one column) ----
const title = [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 0, after: 200 },
    children: [new TextRun({ text: "Reconstructing Sprint-Start Jira State for Leakage-Aware Outcome Prediction", font: FONT, size: 48 })] }),
];
const authors = [
  ["Hani AbuSharkh", "Higher Colleges of Technology", "Al Ain, UAE", "habusharkh@hct.ac.ae"],
  ["Maryam Hanif", "United Arab Emirates University", "Al Ain, UAE", "700052696@uaeu.ac.ae"],
  ["Rafat Damseh", "United Arab Emirates University", "Al Ain, UAE", "rdamseh@uaeu.ac.ae"],
];
const aw = Math.floor(FULL / 3);
title.push(new Table({
  width: { size: aw * 3, type: WidthType.DXA }, columnWidths: [aw, aw, aw], layout: TableLayoutType.FIXED,
  rows: [new TableRow({ children: authors.map((a) => new TableCell({
    width: { size: aw, type: WidthType.DXA },
    borders: { top: noBorder, bottom: noBorder, left: noBorder, right: noBorder },
    children: a.map((line, k) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 0, after: 0 },
      children: [new TextRun({ text: line, font: FONT, size: k === 0 ? 22 : 20, italics: k > 0 && k < 3 })] })),
  })) })],
}));
title.push(new Paragraph({ spacing: { before: 0, after: 200 }, children: [] }));

// ---- body sections ----
const sections = [{ properties: { page: pageProps() }, children: title }];
let cur = [];
let h1 = 0, h2 = 0;
const flush = (cols) => { if (cur.length) sections.push({ properties: { type: SectionType.CONTINUOUS, column: cols ? { count: 2, space: GAP, equalWidth: true } : { count: 1 }, page: pageProps() }, children: cur }); cur = []; };
function pageProps() { return { size: { width: PAGE_W, height: 11 * IN }, margin: { top: 0.75 * IN, bottom: 1 * IN, left: MARGIN_LR, right: MARGIN_LR } }; }

const bodyP = (rs, opts = {}) => new Paragraph({
  alignment: AlignmentType.JUSTIFIED, spacing: { before: 0, after: 0, line: 230 },
  indent: opts.noIndent ? undefined : { firstLine: 200 }, children: rs,
});

let afterHeading = false;
for (const b of blocks) {
  if (b.k === "abstract") {
    cur.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { before: 0, after: 120 },
      children: [new TextRun({ text: "Abstract—", font: FONT, size: 18, bold: true, italics: true }), ...runs(b.runs, 18, { bold: true })] }));
  } else if (b.k === "keywords") {
    cur.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { before: 0, after: 160 },
      children: [new TextRun({ text: "Index Terms—", font: FONT, size: 18, bold: true, italics: true }), ...runs(b.runs, 18, { bold: true, italics: true })] }));
  } else if (b.k === "h1") {
    const label = b.nonum ? b.text.toUpperCase() : `${ROMAN[h1++]}. ${b.text.toUpperCase()}`;
    h2 = 0;
    cur.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 200, after: 100 }, keepNext: true,
      children: [new TextRun({ text: label, font: FONT, size: 20, smallCaps: true })] }));
    afterHeading = true;
  } else if (b.k === "h2") {
    cur.push(new Paragraph({ spacing: { before: 120, after: 60 }, keepNext: true,
      children: [new TextRun({ text: `${"ABCDEFGHIJ"[h2++]}. ${b.text}`, font: FONT, size: 20, italics: true })] }));
    afterHeading = true;
  } else if (b.k === "p") {
    const bold = b.runs[0] && b.runs[0].b;   // threats paragraphs start with a bold label
    cur.push(bodyP(runs(b.runs, 20), { noIndent: false && bold }));
    afterHeading = false;
  } else if (b.k === "table" && b.wide) {
    flush(true); cur.push(...tableBlock(b)); flush(false);
  } else if (b.k === "table") {
    cur.push(...tableBlock(b));
  } else if (b.k === "figure") {
    cur.push(...figureBlock(b));
  } else if (b.k === "ref") {
    cur.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { before: 0, after: 20 },
      indent: { left: 360, hanging: 360 },
      children: [new TextRun({ text: `[${b.num}]\t`, font: FONT, size: 16 }), ...runs(b.runs, 16)] }));
  }
}
flush(true);

const doc = new Document({
  creator: "Hani AbuSharkh", title: "Reconstructing Sprint-Start Jira State for Leakage-Aware Outcome Prediction",
  styles: { default: { document: { run: { font: FONT, size: 20 } } } },
  sections,
});
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(outFile, buf); console.log("wrote", outFile, sections.length, "sections"); });
