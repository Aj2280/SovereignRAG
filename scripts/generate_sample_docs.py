import os
import csv
from docx import Document
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

out_dir = "/Users/abhi/Desktop/rea/rag /documents_to_upload"
os.makedirs(out_dir, exist_ok=True)

# ------------------------------------------------------------------------------
# 1. AI_Infrastructure_Whitepaper_2026.pdf
# ------------------------------------------------------------------------------
pdf_path = os.path.join(out_dir, "AI_Infrastructure_Whitepaper_2026.pdf")
doc = SimpleDocTemplate(pdf_path, pagesize=letter, rightMargin=54, leftMargin=54, topMargin=54, bottomMargin=54)
styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    "DocTitle",
    parent=styles["Heading1"],
    fontSize=20,
    leading=24,
    textColor=colors.HexColor("#1a365d"),
    spaceAfter=10,
)
h2_style = ParagraphStyle(
    "DocH2",
    parent=styles["Heading2"],
    fontSize=13,
    leading=17,
    textColor=colors.HexColor("#2b6cb0"),
    spaceBefore=12,
    spaceAfter=6,
)
body_style = ParagraphStyle(
    "DocBody",
    parent=styles["Normal"],
    fontSize=9.5,
    leading=13.5,
    textColor=colors.HexColor("#2d3748"),
    spaceAfter=6,
)

story = []
story.append(Paragraph("Next-Generation AI Infrastructure & Cluster Architecture", title_style))
story.append(Paragraph("<b>Author:</b> Cloud Computing Systems Group &nbsp;|&nbsp; <b>Published:</b> Q1 2026 &nbsp;|&nbsp; <b>Version:</b> 2.4", body_style))
story.append(Spacer(1, 10))

story.append(Paragraph("1. Executive Overview and Scalability Frontiers", h2_style))
story.append(Paragraph(
    "Modern large-scale foundation models require co-design across silicon, interconnects, and distributed storage. "
    "The cluster architecture detailed herein supports training and inference workloads exceeding 100,000 accelerator units. "
    "By deploying optical circuit switching (OCS) combined with InfiniBand NDR 800 Gbps fabrics, cross-node collective latency "
    "is reduced by 42% compared to standard fat-tree Ethernet implementations.",
    body_style,
))

story.append(Paragraph("2. Accelerator Specifications & Thermal Management", h2_style))
story.append(Paragraph(
    "High thermal design power (TDP) per socket has mandated a complete transition from air cooling to direct-to-chip liquid cooling. "
    "Our generation-4 cooling loops maintain a maximum coolant inlet temperature of 32 degrees Celsius, enabling full PUE efficiency of 1.08.",
    body_style,
))

table_data = [
    ["Cluster Tier", "Accelerator ASIC", "HBM3e Memory", "Interconnect Bandwidth", "Peak FP8 TFLOPS"],
    ["Tier 1 Training", "Custom Tensor-X4", "192 GB per chip", "3.2 Tbps per node", "3,950 TFLOPS"],
    ["Tier 2 Reasoning", "NeuralCore-900", "144 GB per chip", "1.6 Tbps per node", "2,400 TFLOPS"],
    ["Tier 3 Edge Inference", "EdgeSil-M2", "48 GB Unified", "400 Gbps per node", "680 TFLOPS"],
]
t = Table(table_data, colWidths=[105, 105, 95, 115, 85])
t.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2b6cb0")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e0")),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f7fafc"), colors.white]),
]))
story.append(t)
story.append(Spacer(1, 10))

story.append(PageBreak())

story.append(Paragraph("3. Fault-Tolerant Checkpointing and Storage Throughput", h2_style))
story.append(Paragraph(
    "To achieve zero downtime under frequent single-GPU hardware dropouts, asynchronous in-memory checkpointing saves model state "
    "to NVMe-over-Fabrics (NVMe-oF) in under 850 milliseconds. Parallel checkpoint verification is executed concurrently with backward passes, "
    "guaranteeing zero stalled compute bubbles during multi-node checkpoint commits.",
    body_style,
))

story.append(Paragraph("4. Green Energy Sourcing and Carbon Offsets", h2_style))
story.append(Paragraph(
    "All regional compute clusters operate under a 24/7 Carbon-Free Energy (CFE) matching mandate. "
    "Thermal energy rejected from data hall liquid exchangers is redirected into municipal district heating grids, offsetting an estimated "
    "45,000 metric tons of carbon dioxide equivalent annually.",
    body_style,
))

doc.build(story)
print("✓ Created PDF:", pdf_path)


# ------------------------------------------------------------------------------
# 2. Global_Quantum_Computing_Report.docx
# ------------------------------------------------------------------------------
docx_path = os.path.join(out_dir, "Global_Quantum_Computing_Report.docx")
doc_word = Document()

doc_word.add_heading("Global Quantum Computing Progress & Fault-Tolerance Report", level=0)
p_meta = doc_word.add_paragraph("Quantum Research Institute | Annual Strategic Briefing | Classification: Public")
p_meta.runs[0].font.italic = True

doc_word.add_heading("1. Surface Codes and Logical Qubit Scaling", level=1)
doc_word.add_paragraph(
    "The critical milestone for commercial quantum advantage is achieving a logical error rate lower than 1e-6. "
    "Recent breakthroughs using distance-7 surface codes have demonstrated fault-tolerant state preparation and non-destructive syndrome extraction. "
    "Physical qubit counts per dilution refrigerator have scaled from 128 to 1,024 transmon qubits while maintaining average two-qubit gate fidelities of 99.85%."
)

doc_word.add_heading("2. Cryogenic Dilution Infrastructure", level=1)
doc_word.add_paragraph(
    "Operating superconducting quantum processors requires stable base temperatures below 15 milliKelvin. "
    "Pulse-tube cryocoolers with multi-stage closed-cycle Helium-3/Helium-4 dilution refrigerators provide 25 microWatts of cooling power at base plate temperatures."
)

table_word = doc_word.add_table(rows=1, cols=4)
table_word.style = "Light Shading Accent 1"
hdr_cells = table_word.rows[0].cells
hdr_cells[0].text = "Quantum System"
hdr_cells[1].text = "Qubit Modality"
hdr_cells[2].text = "Coherence Time (T2)"
hdr_cells[3].text = "2-Qubit Gate Fidelity"

q_data = [
    ("Helix-Prime", "Superconducting Transmon", "280 microseconds", "99.85%"),
    ("IonTrax-V", "Trapped Ytterbium Ions", "12 seconds", "99.92%"),
    ("PhotonQ-7", "Linear Optical Photons", "Room Temperature", "99.40%"),
    ("SpinGrid-3", "Silicon Spin Qubits", "4.2 milliseconds", "99.70%"),
]

for row in q_data:
    r_cells = table_word.add_row().cells
    for i, val in enumerate(row):
        r_cells[i].text = val

doc_word.add_heading("3. Post-Quantum Cryptography Migration Roadmaps", level=1)
doc_word.add_paragraph(
    "Organizations must transition from classical RSA-2048 and ECC public key infrastructure to lattice-based post-quantum cryptography algorithms. "
    "The NIST standards ML-KEM (Kyber) for key encapsulation and ML-DSA (Dilithium) for digital signatures are mandated for complete compliance by Q4 2027."
)

doc_word.save(docx_path)
print("✓ Created DOCX:", docx_path)


# ------------------------------------------------------------------------------
# 3. Enterprise_Security_Compliance_2026.md
# ------------------------------------------------------------------------------
md_path = os.path.join(out_dir, "Enterprise_Security_Compliance_2026.md")
md_text = """# Enterprise Security & Compliance Governance Policy (2026)

**Document Control:** POL-SEC-2026-V3  
**Effective Date:** January 1, 2026  
**Review Cycle:** Semi-Annual  
**Target Audience:** Engineering, SecOps, DevSecOps, and Legal

---

## Section 1: Data Encryption at Rest and in Transit

All sensitive data assets, relational databases, vector embeddings, and object storage volumes must comply with the following cryptographic standards:

1. **Encryption at Rest:**
   - Primary cipher: AES-256-GCM with hardware-backed KMS (Envelope Encryption).
   - Key rotation cycle: Cryptographic keys must automatically rotate every 90 days.
   - Quantum resilience: Data stored with retention >5 years must utilize hybrid Post-Quantum Kyber-768 key encapsulation.

2. **Encryption in Transit:**
   - Mandatory protocol: TLS 1.3 only; legacy TLS 1.0, 1.1, and 1.2 are blocked at all ingress proxies.
   - Cipher suites: TLS_AES_256_GCM_SHA384 and TLS_CHACHA20_POLY1305_SHA256.
   - Mutual TLS (mTLS): Required for all service-to-service microservice communications.

---

## Section 2: Identity, Access Management (IAM), and Zero Trust

The organization operates under a strict Zero Trust Architecture (ZTA) principle: "Never trust, always verify."

- **Multi-Factor Authentication (MFA):** FIDO2 / WebAuthn hardware security keys (such as YubiKey) are mandatory for all employee console and shell access. SMS and phone apps are strictly deprecated.
- **Principle of Least Privilege (PoLP):** Role-Based Access Control (RBAC) grants ephemeral credentials with a maximum Time-To-Live (TTL) of 60 minutes.
- **Service Accounts:** Hardcoded API tokens in configuration files or code repositories are forbidden; all machines must authenticate using Workload Identity Federation (WIF).

---

## Section 3: Incident Response Severity Levels and SLAs

Security events are categorized into four operational tiers with mandatory response SLAs:

| Severity Level | Definition | First Response SLA | Escalation Window | Target Resolution |
| :--- | :--- | :--- | :--- | :--- |
| **SEV-1 (Critical)** | Active data exfiltration, total outage, or root compromise | < 15 minutes | Immediate (CISO & VP) | < 4 hours |
| **SEV-2 (High)** | Vulnerability exploited, partial service disruption | < 30 minutes | 1 hour (SecOps Lead) | < 12 hours |
| **SEV-3 (Medium)** | Non-exploited critical CVE, abnormal telemetry anomaly | < 2 hours | 4 hours (Team Lead) | < 48 hours |
| **SEV-4 (Low)** | Minor misconfiguration, policy hygiene warning | < 24 hours | Next sprint review | < 7 business days |

---

## Section 4: Audit Logging and Immutable SIEM Retention

Audit records must be streamed in real-time to write-once-read-many (WORM) cloud object storage with a retention guarantee of no less than 7 years. All log streams are cryptographically signed using SHA-384 Merkle trees to detect tampering or log omission.
"""
with open(md_path, "w", encoding="utf-8") as f:
    f.write(md_text)
print("✓ Created Markdown:", md_path)


# ------------------------------------------------------------------------------
# 4. Quarterly_Financial_Performance.csv
# ------------------------------------------------------------------------------
csv_path = os.path.join(out_dir, "Quarterly_Financial_Performance.csv")
csv_data = [
    ["Quarter", "Division", "Revenue_USD_Millions", "RnD_Expenditure_Millions", "Gross_Margin_Percent", "Operating_Income_Millions", "Headcount"],
    ["Q1 2025", "Cloud AI Systems", "420.5", "125.0", "74.2%", "142.3", "1420"],
    ["Q1 2025", "Enterprise Software", "310.2", "62.4", "81.5%", "118.0", "980"],
    ["Q1 2025", "Cybersecurity Ops", "185.0", "41.2", "78.0", "58.5", "620"],
    ["Q2 2025", "Cloud AI Systems", "495.8", "140.2", "75.0%", "171.4", "1550"],
    ["Q2 2025", "Enterprise Software", "325.4", "64.0", "82.1%", "124.6", "1010"],
    ["Q2 2025", "Cybersecurity Ops", "202.1", "44.5", "78.5%", "65.2", "650"],
    ["Q3 2025", "Cloud AI Systems", "580.3", "162.0", "76.1%", "208.9", "1720"],
    ["Q3 2025", "Enterprise Software", "340.0", "66.1", "82.4%", "131.2", "1040"],
    ["Q3 2025", "Cybersecurity Ops", "224.6", "48.0", "79.2%", "74.1", "680"],
    ["Q4 2025", "Cloud AI Systems", "695.0", "190.5", "77.5%", "258.0", "1910"],
    ["Q4 2025", "Enterprise Software", "365.8", "69.0", "83.0%", "142.5", "1080"],
    ["Q4 2025", "Cybersecurity Ops", "250.4", "52.3", "80.1%", "86.4", "720"],
]
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(csv_data)
print("✓ Created CSV:", csv_path)

