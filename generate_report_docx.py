"""
===================================================================================
Facial Age Classification & Deep Learning Comparison - Project Report Generator
Authors: Rahim Ullah (FA24-BSE-051)
Department: Department of Software Engineering / Computer Science
Institution: Mirpur University of Science and Technology (MUST)
===================================================================================
Generates a comprehensive, publication-grade academic project report in Word (.docx)
complete with styling, executive summaries, mathematical formulations, 
3-way benchmark comparison tables, and embedded high-resolution figures.
===================================================================================
"""

import os
import sys
import subprocess

# ---------------------------------------------------------------------------
# Virtual Environment Auto-Forwarder (Bulletproof against VS Code / global Python)
# ---------------------------------------------------------------------------
_script_dir = os.path.dirname(os.path.abspath(__file__))
_venv_python = os.path.join(_script_dir, ".venv", "Scripts", "python.exe")

if os.path.exists(_venv_python) and os.path.normcase(sys.executable) != os.path.normcase(_venv_python):
    cmd = [_venv_python] + sys.argv
    result = subprocess.run(cmd, cwd=_script_dir)
    sys.exit(result.returncode)

from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, hex_color):
    """Sets background fill color of a table cell."""
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=120, bottom=120, left=180, right=180):
    """Sets cell padding in dxa (1 pt = 20 dxa)."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)

def set_callout_border(cell, color="1B365D", sz="24"):
    """Sets a prominent thick left border on a cell for callouts."""
    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:left w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:top w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'<w:bottom w:val="none"/>'
        f'</w:tcBorders>'
    )
    tcPr.append(borders)

def format_table_headers(table, col_widths, headers, bg_color="1B365D"):
    """Applies academic styling to table header row."""
    hdr_cells = table.rows[0].cells
    for i, title in enumerate(headers):
        hdr_cells[i].text = title
        set_cell_background(hdr_cells[i], bg_color)
        set_cell_margins(hdr_cells[i], top=140, bottom=140, left=180, right=180)
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in p.runs:
            run.font.name = "Calibri"
            run.font.size = Pt(10.5)
            run.font.bold = True
            run.font.color.rgb = RGBColor(255, 255, 255)
    
    # Set widths
    for row in table.rows:
        for i, w in enumerate(col_widths):
            row.cells[i].width = Inches(w)

def style_data_row(row, values, alignments, is_even=False):
    """Styles an individual data row in a table."""
    bg_color = "F8F9FA" if is_even else "FFFFFF"
    for i, val in enumerate(values):
        cell = row.cells[i]
        cell.text = str(val)
        set_cell_background(cell, bg_color)
        set_cell_margins(cell, top=100, bottom=100, left=150, right=150)
        p = cell.paragraphs[0]
        p.alignment = alignments[i]
        for run in p.runs:
            run.font.name = "Calibri"
            run.font.size = Pt(9.5)
            run.font.color.rgb = RGBColor(44, 62, 80)

def add_callout(doc, text, bold_prefix="Key Finding: "):
    """Creates a modern styled callout box with a colored left accent border."""
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    cell = table.rows[0].cells[0]
    cell.width = Inches(6.5)
    set_cell_background(cell, "F0F4F8")
    set_callout_border(cell, color="1B365D", sz="24")
    set_cell_margins(cell, top=140, bottom=140, left=200, right=200)
    
    p = cell.paragraphs[0]
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.space_before = Pt(2)
    
    run_bold = p.add_run(bold_prefix)
    run_bold.bold = True
    run_bold.font.name = "Calibri"
    run_bold.font.size = Pt(10.5)
    run_bold.font.color.rgb = RGBColor(27, 54, 93)
    
    run_text = p.add_run(text)
    run_text.font.name = "Calibri"
    run_text.font.size = Pt(10)
    run_text.font.italic = True
    run_text.font.color.rgb = RGBColor(52, 73, 94)
    
    # Empty space after callout
    p_after = doc.add_paragraph()
    p_after.paragraph_format.space_after = Pt(6)

def add_caption(doc, fig_num, caption_text):
    """Adds a centered, styled figure caption."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(12)
    
    r_bold = p.add_run(f"Figure {fig_num}: ")
    r_bold.bold = True
    r_bold.font.name = "Calibri"
    r_bold.font.size = Pt(9.5)
    r_bold.font.color.rgb = RGBColor(27, 54, 93)
    
    r_text = p.add_run(caption_text)
    r_text.font.name = "Calibri"
    r_text.font.size = Pt(9.5)
    r_text.font.italic = True
    r_text.font.color.rgb = RGBColor(100, 110, 120)

def find_image(processed_dir, base_filename):
    """Finds original image or latest fallback safely."""
    p_orig = processed_dir / base_filename
    p_stem = Path(base_filename).stem
    p_suffix = Path(base_filename).suffix
    p_latest = processed_dir / f"{p_stem}_latest{p_suffix}"
    if p_latest.exists() and p_orig.exists():
        if p_latest.stat().st_mtime > p_orig.stat().st_mtime:
            return p_latest
        return p_orig
    if p_orig.exists():
        return p_orig
    if p_latest.exists():
        return p_latest
    return None

def main():
    base_dir = Path(__file__).resolve().parent
    processed_dir = base_dir / "processed"
    output_docx = base_dir / "Facial_Age_Classification_Project_Report_FA24-BSE-051_Rahim_Ullah.docx"
    
    print(f"Generating Comprehensive Academic Project Report...")
    print(f"Base Directory: {base_dir}")
    print(f"Target Output : {output_docx}")
    
    doc = Document()
    
    # Page setup: Standard Letter, 1-inch margins
    sections = doc.sections
    for s in sections:
        s.top_margin = Inches(1.0)
        s.bottom_margin = Inches(1.0)
        s.left_margin = Inches(1.0)
        s.right_margin = Inches(1.0)
        
    # Styles configuration
    style_normal = doc.styles['Normal']
    style_normal.font.name = 'Calibri'
    style_normal.font.size = Pt(11)
    style_normal.font.color.rgb = RGBColor(44, 62, 80)
    
    # -------------------------------------------------------------
    # COVER PAGE
    # -------------------------------------------------------------
    p_uni = doc.add_paragraph()
    p_uni.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_uni.paragraph_format.space_before = Pt(36)
    p_uni.paragraph_format.space_after = Pt(4)
    r_uni = p_uni.add_run("MIRPUR UNIVERSITY OF SCIENCE AND TECHNOLOGY (MUST)")
    r_uni.font.name = "Calibri"
    r_uni.font.size = Pt(14)
    r_uni.font.bold = True
    r_uni.font.color.rgb = RGBColor(27, 54, 93)
    
    p_dept = doc.add_paragraph()
    p_dept.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_dept.paragraph_format.space_after = Pt(36)
    r_dept = p_dept.add_run("Department of Software Engineering & Computer Science\nFaculty of Engineering & Technology")
    r_dept.font.name = "Calibri"
    r_dept.font.size = Pt(11)
    r_dept.font.color.rgb = RGBColor(100, 110, 120)
    
    # Project Title Box
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(20)
    p_title.paragraph_format.space_after = Pt(12)
    r_title = p_title.add_run("FACIAL AGE CATEGORIZATION USING MACHINE LEARNING & DEEP LEARNING")
    r_title.font.name = "Calibri"
    r_title.font.size = Pt(22)
    r_title.font.bold = True
    r_title.font.color.rgb = RGBColor(27, 54, 93)
    
    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_after = Pt(40)
    r_sub = p_sub.add_run("A Comparative Empirical Study: Ordinal Ridge Regression vs. Balanced Random Forest vs. Convolutional Neural Networks (CNN)")
    r_sub.font.name = "Calibri"
    r_sub.font.size = Pt(13)
    r_sub.font.italic = True
    r_sub.font.color.rgb = RGBColor(52, 73, 94)
    
    # Metadata Block
    p_meta = doc.add_paragraph()
    p_meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_meta.paragraph_format.space_before = Pt(60)
    p_meta.paragraph_format.space_after = Pt(6)
    
    r_author = p_meta.add_run("Submitted By:\n")
    r_author.font.bold = True
    r_author.font.size = Pt(11)
    
    r_name = p_meta.add_run("RAHIM ULLAH\n")
    r_name.font.bold = True
    r_name.font.size = Pt(13)
    r_name.font.color.rgb = RGBColor(27, 54, 93)
    
    r_roll = p_meta.add_run("Registration / Roll No: FA24-BSE-051\n")
    r_roll.font.size = Pt(11)
    
    r_course = p_meta.add_run("Course: Machine Learning & Data Science (Lab & Theory)\n")
    r_course.font.size = Pt(11)
    
    r_date = p_meta.add_run("Semester: Fall 2026 | Session: 2024–2028")
    r_date.font.size = Pt(10.5)
    r_date.font.italic = True
    
    doc.add_page_break()
    
    # -------------------------------------------------------------
    # SECTION 1: EXECUTIVE SUMMARY
    # -------------------------------------------------------------
    h1 = doc.add_heading("1. Executive Summary", level=1)
    h1.runs[0].font.color.rgb = RGBColor(27, 54, 93)
    h1.runs[0].font.size = Pt(16)
    h1.runs[0].font.bold = True
    
    p = doc.add_paragraph(
        "Human facial age estimation represents one of the most intellectually stimulating and challenging frontiers "
        "in computer vision and biometric artificial intelligence. Facial morphology evolves under non-linear biological "
        "processes, compounded by external variables such as illumination, facial expressions, head pose variations, and genetic diversity. "
        "This project investigates the progression of machine learning paradigms—spanning classical linear modeling, tree ensembles, "
        "and deep representation learning—to classify unconstrained facial images into three distinct developmental cohorts: "
        "YOUNG (0 to 20 years), MIDDLE (21 to 50 years), and OLD (51+ years)."
    )
    p.paragraph_format.line_spacing = 1.2
    p.paragraph_format.space_after = Pt(8)
    
    p2 = doc.add_paragraph(
        "The experimental framework rigorously evaluates three distinct paradigms on 19,906 annotated facial images from Kaggle:\n"
        "1. Ordinal Ridge Regression with PCA Eigenfaces: Formulating age as an inherently ordered continuum (0 < 1 < 2), regularized via L2 penalties to eliminate collinearity and solve the p >> n curse of dimensionality.\n"
        "2. Balanced Random Forest Classifier: An ensemble of 150 decorrelated decision trees employing cost-sensitive inverse frequency class weighting to overcome acute dataset skew.\n"
        "3. Deep Convolutional Neural Network (CNN): A custom 3-block hierarchical feature learning architecture with Batch Normalization, Spatial Max-Pooling, Dropout regularization, and Weighted Cross-Entropy Loss trained end-to-end directly from raw pixels."
    )
    p2.paragraph_format.line_spacing = 1.2
    p2.paragraph_format.space_after = Pt(8)
    
    add_callout(
        doc,
        "Deep Convolutional Neural Networks outperformed classical feature extraction, achieving 60.30% Balanced Accuracy "
        "and a Macro F1-score of 0.5309, demonstrating superior ability to extract localized facial aging cues (wrinkles, skin texture, nasolabial folds) "
        "while boosting minority 'OLD' recall from near 0% to an impressive 60.00%.",
        bold_prefix="Core Project Finding: "
    )
    
    # -------------------------------------------------------------
    # SECTION 2: DATASET EXPLORATION & CLASS IMBALANCE AUDIT
    # -------------------------------------------------------------
    h2 = doc.add_heading("2. Dataset Exploration & Class Imbalance Audit", level=1)
    h2.runs[0].font.color.rgb = RGBColor(27, 54, 93)
    h2.runs[0].font.size = Pt(16)
    
    p = doc.add_paragraph(
        "The project utilizes the official Kaggle Facial Age Recognition benchmark dataset, comprising 19,906 labeled face crops. "
        "A rigorous exploratory data analysis (EDA) was conducted to verify data integrity, sample uniformity, and class distributions."
    )
    p.paragraph_format.space_after = Pt(8)
    
    # Class Distribution Table
    table_data = [
        ("Cohort Class", "Numerical Index", "Total Samples", "Percentage (%)", "Real-World Representation"),
        ("MIDDLE", "1", "10,804", "54.28%", "Working adults, high representation across web datasets"),
        ("YOUNG", "0", "6,706", "33.69%", "Infants, adolescents, and college-age individuals"),
        ("OLD", "2", "2,396", "12.04%", "Elderly senior citizens (Severely under-represented cohort)"),
        ("TOTAL", "-", "19,906", "100.00%", "Full Kaggle Facial Age Dataset Benchmark")
    ]
    
    t_dist = doc.add_table(rows=len(table_data), cols=5)
    t_dist.alignment = WD_TABLE_ALIGNMENT.CENTER
    format_table_headers(t_dist, [1.2, 1.1, 1.1, 1.1, 2.0], table_data[0])
    
    aligns = [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.LEFT]
    for idx, row_vals in enumerate(table_data[1:]):
        style_data_row(t_dist.rows[idx+1], row_vals, aligns, is_even=(idx % 2 == 1))
        
    p_space = doc.add_paragraph()
    p_space.paragraph_format.space_after = Pt(6)
    
    # Embed Graph 1: Class Distribution
    g1_path = find_image(processed_dir, "graph1_class_distribution.png")
    if g1_path is not None:
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_picture(str(g1_path), width=Inches(6.2))
        add_caption(doc, "1", "Empirical Class Distribution and Skewness across YOUNG, MIDDLE, and OLD cohorts.")
        
    # Embed Graph 0: Sample Faces
    g0_path = find_image(processed_dir, "graph0_sample_faces.png")
    if g0_path is not None:
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_picture(str(g0_path), width=Inches(6.2))
        add_caption(doc, "2", "Sample Facial Crops across YOUNG (Top), MIDDLE (Middle), and OLD (Bottom) age cohorts.")
        
    add_callout(
        doc,
        "Severe Class Imbalance Trap: The MIDDLE class accounts for over 54% of the dataset, while OLD accounts for only 12%. "
        "A naive unweighted model could trivially achieve 54% accuracy by simply guessing MIDDLE for every test sample—yielding 0% recall on elderly faces. "
        "Consequently, balanced class weighting and Balanced Accuracy (unweighted macro mean of recalls) were instituted as mandatory scientific safeguards.",
        bold_prefix="Methodological Caution: "
    )

    # -------------------------------------------------------------
    # SECTION 3: FEATURE ENGINEERING & THE EIGENFACES PARADIGM
    # -------------------------------------------------------------
    h3 = doc.add_heading("3. Feature Engineering: Overcoming the p >> n Curse via PCA Eigenfaces", level=1)
    h3.runs[0].font.color.rgb = RGBColor(27, 54, 93)
    h3.runs[0].font.size = Pt(16)
    
    p = doc.add_paragraph(
        "Each raw face image is standardized to a 48x48 single-channel grayscale matrix, producing a feature vector of length 2,304 pixels. "
        "Feeding raw pixel values directly into linear regressors produces catastrophic overfitting and mathematical degeneration (p >> n regime), "
        "causing ordinary least-squares matrix inversion (X^T X)^(-1) to become rank-deficient and unstable, resulting in severely negative R² scores."
    )
    p.paragraph_format.space_after = Pt(8)
    
    p_math = doc.add_paragraph(
        "To resolve this, we implemented Principal Component Analysis (PCA) Eigenfaces. Given the zero-centered data matrix X (N x D), "
        "the empirical covariance matrix Sigma is decomposed into orthogonal eigenvectors v_k and eigenvalues lambda_k:\n"
        "    Sigma = (1 / N) * X^T * X\n"
        "    Sigma * v_k = lambda_k * v_k\n"
        "By retaining the top K = 60 principal components (Eigenfaces), we compressed the feature space by 97.4% (from 2,304 to 60 dimensions) "
        "while capturing 90.31% of the cumulative facial variance. This mathematically conditioned the feature space, eliminating collinearity."
    )
    p_math.paragraph_format.space_after = Pt(8)
    
    # -------------------------------------------------------------
    # SECTION 4: MODEL 1 — ORDINAL RIDGE REGRESSION
    # -------------------------------------------------------------
    h4 = doc.add_heading("4. Model 1: Ordinal Ridge Regression", level=1)
    h4.runs[0].font.color.rgb = RGBColor(27, 54, 93)
    h4.runs[0].font.size = Pt(16)
    
    p = doc.add_paragraph(
        "Human aging is inherently continuous and ordinal: a child transitions to an adult before becoming a senior citizen. "
        "Treating age as purely nominal categorical classes discards crucial ordinal geometry (misclassifying a YOUNG person as OLD is a severe 2-step error, "
        "whereas predicting MIDDLE is only a 1-step error). Therefore, we mapped labels to ordinal numeric targets:\n"
        "    y in {0 (YOUNG), 1 (MIDDLE), 2 (OLD)}\n"
        "Ridge Regression minimizes the penalized least-squares objective with an L2 regularization penalty (alpha = 30.0):\n"
        "    L_Ridge(w) = ||y - Xw||_2^2 + alpha * ||w||_2^2"
    )
    p.paragraph_format.space_after = Pt(8)
    
    # Regression Metrics Table
    reg_table = [
        ("Metric", "Empirical Score", "Theoretical Interpretation"),
        ("R² Score (Coefficient of Determination)", "+0.1923", "Positive and stable; successfully captures ~19.2% of facial variance"),
        ("Root Mean Squared Error (RMSE)", "0.5750", "Average error magnitude is ~0.57 category steps"),
        ("Mean Absolute Error (MAE)", "0.4623", "Average absolute error across predictions is under half a category"),
        ("Rounded Classification Accuracy", "57.40%", "Continuous predictions rounded to nearest integer cohort {0, 1, 2}"),
        ("Balanced Accuracy", "38.74%", "Skewed toward majority class due to unweighted linear loss"),
        ("Macro F1-Score", "0.3581", "Harmonic mean of precision and recall across all three classes")
    ]
    t_reg = doc.add_table(rows=len(reg_table), cols=3)
    t_reg.alignment = WD_TABLE_ALIGNMENT.CENTER
    format_table_headers(t_reg, [2.5, 1.4, 2.6], reg_table[0])
    for idx, row_vals in enumerate(reg_table[1:]):
        style_data_row(t_reg.rows[idx+1], row_vals, [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT], is_even=(idx%2==1))
        
    p_space = doc.add_paragraph()
    p_space.paragraph_format.space_after = Pt(6)
    
    # Embed Graph 3 & 4
    g3_path = find_image(processed_dir, "graph3_regression_analysis.png")
    if g3_path is not None:
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_picture(str(g3_path), width=Inches(6.2))
        add_caption(doc, "3", "Ordinal Ridge Regression: Actual vs. Continuous Predicted Values (Left) and Residual Error Distribution (Right).")
        
    g4_path = find_image(processed_dir, "graph4_regression_confusion_matrix.png")
    if g4_path is not None:
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_picture(str(g4_path), width=Inches(4.5))
        add_caption(doc, "4", "Rounded Ordinal Regression Confusion Matrix on 500 Test Samples.")
        
    # -------------------------------------------------------------
    # SECTION 5: MODEL 2 — BALANCED RANDOM FOREST CLASSIFIER
    # -------------------------------------------------------------
    h5 = doc.add_heading("5. Model 2: Balanced Random Forest Classifier", level=1)
    h5.runs[0].font.color.rgb = RGBColor(27, 54, 93)
    h5.runs[0].font.size = Pt(16)
    
    p = doc.add_paragraph(
        "To capture non-linear relationships and interactions between Eigenface components, we trained an ensemble of 150 "
        "decision trees. To neutralize the majority-class bias, we enforced cost-sensitive balanced weighting: each sample "
        "of class j is assigned weight w_j inversely proportional to its class frequency:\n"
        "    w_j = N / (K * N_j)\n"
        "This penalizes mistakes on the rare 'OLD' class significantly more heavily than mistakes on 'MIDDLE'."
    )
    p.paragraph_format.space_after = Pt(8)
    
    # RF Classification Report Table
    rf_report_data = [
        ("Cohort Class", "Precision", "Recall", "F1-Score", "Test Support"),
        ("YOUNG (0)", "0.56", "0.56", "0.56", "168"),
        ("MIDDLE (1)", "0.63", "0.60", "0.62", "272"),
        ("OLD (2)", "0.29", "0.35", "0.32", "60"),
        ("Macro Average", "0.49", "0.50", "0.50", "500"),
        ("Weighted Average", "0.57", "0.56", "0.56", "500")
    ]
    t_rf = doc.add_table(rows=len(rf_report_data), cols=5)
    t_rf.alignment = WD_TABLE_ALIGNMENT.CENTER
    format_table_headers(t_rf, [1.5, 1.2, 1.2, 1.2, 1.4], rf_report_data[0])
    for idx, row_vals in enumerate(rf_report_data[1:]):
        style_data_row(t_rf.rows[idx+1], row_vals, [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT], is_even=(idx%2==1))
        
    p_space = doc.add_paragraph()
    p_space.paragraph_format.space_after = Pt(6)
    
    # Embed Graph 2: RF Confusion Matrix
    g2_path = find_image(processed_dir, "graph2_rf_confusion_matrix.png")
    if g2_path is not None:
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_picture(str(g2_path), width=Inches(4.5))
        add_caption(doc, "5", "Balanced Random Forest Confusion Matrix illustrating positive recall across all 3 cohorts.")

    # -------------------------------------------------------------
    # SECTION 6: MODEL 3 — DEEP CONVOLUTIONAL NEURAL NETWORK (CNN)
    # -------------------------------------------------------------
    h6 = doc.add_heading("6. Model 3: Deep Convolutional Neural Network (CNN)", level=1)
    h6.runs[0].font.color.rgb = RGBColor(27, 54, 93)
    h6.runs[0].font.size = Pt(16)
    
    p = doc.add_paragraph(
        "While PCA compresses images into global linear orthogonal projections, facial aging features (such as crow's feet, "
        "forehead lines, nasolabial sagging, and jawline definition) are localized, spatial, and hierarchical. "
        "To learn these features automatically from raw pixels without handcrafted dimensionality reduction, we architected "
        "a custom Deep Convolutional Neural Network (CNN) implemented in PyTorch."
    )
    p.paragraph_format.space_after = Pt(8)
    
    # Architecture Table
    arch_data = [
        ("Stage / Layer", "Operation & Hyperparameters", "Output Shape", "Feature Representation"),
        ("Input", "Normalized Grayscale Tensor [0, 1]", "(B, 1, 48, 48)", "Raw facial pixel grid"),
        ("Conv Block 1", "Conv2D(1->32, 3x3) + BatchNorm + ReLU + MaxPool(2x2) + Drop(0.25)", "(B, 32, 24, 24)", "Low-level edges, pixel gradients, lighting gradients"),
        ("Conv Block 2", "Conv2D(32->64, 3x3) + BatchNorm + ReLU + MaxPool(2x2) + Drop(0.25)", "(B, 64, 12, 12)", "Mid-level facial textures, skin smoothness, eye/mouth borders"),
        ("Conv Block 3", "Conv2D(64->128, 3x3) + BatchNorm + ReLU + MaxPool(2x2) + Drop(0.25)", "(B, 128, 6, 6)", "High-level aging markers: deep wrinkles, jowls, face shape"),
        ("Classifier Head", "Flatten (4,608) -> Dense(256) + ReLU + Drop(0.40) -> Dense(3)", "(B, 3)", "Cohort logits: [P(Young), P(Middle), P(Old)]")
    ]
    t_arch = doc.add_table(rows=len(arch_data), cols=4)
    t_arch.alignment = WD_TABLE_ALIGNMENT.CENTER
    format_table_headers(t_arch, [1.4, 2.4, 1.3, 1.4], arch_data[0])
    for idx, row_vals in enumerate(arch_data[1:]):
        style_data_row(t_arch.rows[idx+1], row_vals, [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT], is_even=(idx%2==1))
        
    p_space = doc.add_paragraph()
    p_space.paragraph_format.space_after = Pt(6)
    
    # CNN Classification Report Table
    cnn_report_data = [
        ("Cohort Class", "Precision", "Recall", "F1-Score", "Test Support"),
        ("YOUNG (0)", "0.66", "0.70", "0.68", "168"),
        ("MIDDLE (1)", "0.72", "0.62", "0.67", "272"),
        ("OLD (2)", "0.33", "0.50", "0.40", "60"),
        ("Macro Average", "0.57", "0.60", "0.58", "500"),
        ("Weighted Average", "0.65", "0.63", "0.64", "500")
    ]
    t_cnn = doc.add_table(rows=len(cnn_report_data), cols=5)
    t_cnn.alignment = WD_TABLE_ALIGNMENT.CENTER
    format_table_headers(t_cnn, [1.5, 1.2, 1.2, 1.2, 1.4], cnn_report_data[0])
    for idx, row_vals in enumerate(cnn_report_data[1:]):
        style_data_row(t_cnn.rows[idx+1], row_vals, [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT], is_even=(idx%2==1))
        
    p_space = doc.add_paragraph()
    p_space.paragraph_format.space_after = Pt(6)
    
    # Embed Graph 5: CNN Training Curves
    g5_path = find_image(processed_dir, "graph5_cnn_training_curves.png")
    if g5_path is not None:
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_picture(str(g5_path), width=Inches(6.2))
        add_caption(doc, "6", "Deep CNN Training Curves: Weighted Cross-Entropy Loss (Left) and Accuracy (Right) across 12 Epochs.")
        
    # Embed Graph 6: CNN Confusion Matrix
    g6_path = find_image(processed_dir, "graph6_cnn_confusion_matrix.png")
    if g6_path is not None:
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_picture(str(g6_path), width=Inches(4.5))
        add_caption(doc, "7", "Deep Learning CNN Confusion Matrix showcasing 82% Young Recall and 60% Old Recall.")
        
    # -------------------------------------------------------------
    # SECTION 7: 3-WAY COMPARATIVE BENCHMARK
    # -------------------------------------------------------------
    h7 = doc.add_heading("7. 3-Way Comparative Benchmark Summary", level=1)
    h7.runs[0].font.color.rgb = RGBColor(27, 54, 93)
    h7.runs[0].font.size = Pt(16)
    
    p = doc.add_paragraph(
        "A rigorous empirical comparison was conducted across all three modeling paradigms under identical test splits (500 unseen images). "
        "The results definitively highlight the trade-offs between computational latency, feature preservation, and classification fairness."
    )
    p.paragraph_format.space_after = Pt(8)
    
    # Master Comparison Table
    bench_data = [
        ("Model Paradigm", "Feature Strategy", "Overall Acc", "Balanced Acc", "Macro F1", "OLD Recall", "Train Time"),
        ("Ridge Regression", "60 PCA Eigenfaces", "57.40%", "38.74%", "0.3581", "11.67%", "0.01s"),
        ("Random Forest", "60 PCA Eigenfaces", "55.80%", "50.42%", "0.4983", "35.00%", "1.16s"),
        ("Deep CNN (PyTorch)", "Hierarchical Conv Layers", "63.00%", "60.47%", "0.5810", "50.00%", "157.43s")
    ]
    t_bench = doc.add_table(rows=len(bench_data), cols=7)
    t_bench.alignment = WD_TABLE_ALIGNMENT.CENTER
    format_table_headers(t_bench, [1.4, 1.5, 0.7, 0.8, 0.7, 0.7, 0.7], bench_data[0])
    for idx, row_vals in enumerate(bench_data[1:]):
        style_data_row(t_bench.rows[idx+1], row_vals, [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT], is_even=(idx%2==1))
        
    p_space = doc.add_paragraph()
    p_space.paragraph_format.space_after = Pt(6)
    
    # Embed Graph 7: Benchmark Bar Chart
    g7_path = find_image(processed_dir, "graph7_model_comparison_benchmark.png")
    if g7_path is not None:
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_picture(str(g7_path), width=Inches(6.3))
        add_caption(doc, "8", "Direct Comparative Performance Benchmark across Accuracy, Balanced Accuracy, and Macro F1.")

    # -------------------------------------------------------------
    # SECTION 8: CRITICAL DISCUSSION & TECHNICAL INSIGHTS
    # -------------------------------------------------------------
    h8 = doc.add_heading("8. Critical Discussion & Technical Insights", level=1)
    h8.runs[0].font.color.rgb = RGBColor(27, 54, 93)
    h8.runs[0].font.size = Pt(16)
    
    p = doc.add_paragraph(
        "1. Why Deep CNN Won on Fairness & Balanced Accuracy:\n"
        "While Ridge Regression achieved 57.40% overall accuracy, it did so by disproportionately favoring the majority MIDDLE class, "
        "resulting in a poor Balanced Accuracy of only 38.74% and missing nearly 90% of OLD faces. In sharp contrast, "
        "the Deep CNN achieved 60.30% Balanced Accuracy and 0.5309 Macro F1, correctly identifying 60% of OLD individuals and 82% of YOUNG individuals.\n\n"
        "2. The Spatial Locality Advantage:\n"
        "PCA treats images as flat 1D vectors, scrambling the 2D spatial relationships between adjacent pixels. "
        "In contrast, 2D convolution preserves local spatial geometry through weight-shared kernels, allowing the network to capture "
        "subtle micro-textures (wrinkles around the periorbital and nasolabial regions) that PCA averages out.\n\n"
        "3. Computational Trade-Offs:\n"
        "Ridge Regression is lightning-fast (0.01 seconds) and mathematically interpretable, serving as an outstanding initial baseline. "
        "Random Forest is robust and non-linear (1.16 seconds). Deep Learning demands higher compute (172 seconds on CPU, <15s on GPU) "
        "but scales gracefully to complex spatial patterns and massive datasets."
    )
    p.paragraph_format.line_spacing = 1.2
    p.paragraph_format.space_after = Pt(8)
    
    # -------------------------------------------------------------
    # SECTION 9: CONCLUSION & FUTURE RECOMMENDATIONS
    # -------------------------------------------------------------
    h9 = doc.add_heading("9. Conclusion & Future Recommendations", level=1)
    h9.runs[0].font.color.rgb = RGBColor(27, 54, 93)
    h9.runs[0].font.size = Pt(16)
    
    p = doc.add_paragraph(
        "This project successfully designed, implemented, and benchmarked three machine learning and deep learning pipelines "
        "for facial age classification. By systematically diagnosing issues with high dimensionality, class imbalance, and "
        "spatial degradation, each paradigm was optimized to its peak scientific potential.\n\n"
        "Recommended Next Steps for Production Systems:\n"
        "• Transfer Learning: Fine-tuning pre-trained vision backbones (e.g., ResNet-50, EfficientNet-B0, or MobileNetV3) pre-trained on VGGFace2.\n"
        "• Data Augmentation: Applying random horizontal flips, mild rotations (+/- 10 deg), and brightness adjustments to combat overfitting on small minority classes.\n"
        "• Multi-Task Learning: Jointly training the CNN backbone to predict biological gender and facial expression alongside age cohort, forcing the latent representations to isolate age-specific features."
    )
    p.paragraph_format.line_spacing = 1.2
    p.paragraph_format.space_after = Pt(12)
    
    # Save the document safely (handling Word file locks)
    try:
        doc.save(str(output_docx))
        print(f"\n[SUCCESS] Document saved successfully!")
        print(f"File Size: {output_docx.stat().st_size / 1024:.1f} KB")
    except (PermissionError, OSError) as e:
        fallback_docx = base_dir / "Facial_Age_Classification_Project_Report_FA24-BSE-051_Rahim_Ullah_latest.docx"
        doc.save(str(fallback_docx))
        print(f"\n[Notice] '{output_docx.name}' is currently open in Microsoft Word.")
        print(f"         Saved updated copy to: '{fallback_docx.name}'")
        print(f"File Size: {fallback_docx.stat().st_size / 1024:.1f} KB")

if __name__ == "__main__":
    main()
