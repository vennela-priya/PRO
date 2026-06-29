from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "BrainTumorAI_Project_Report.docx"
BACKGROUND_IMG = ROOT / "BrainTumorAI_Background_and_Motivation.png"


def shade_cell(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def set_cell(cell, text, bold=False, color="000000", size=10):
    cell.text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(0)
    r = p.add_run(text)
    r.bold = bold
    r.font.name = "Times New Roman"
    r.font.size = Pt(size)
    r.font.color.rgb = RGBColor.from_string(color)


def add_para(doc, text="", align=None, bold=False, size=12, space_after=8):
    p = doc.add_paragraph()
    if align:
        p.alignment = align
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.15
    r = p.add_run(text)
    r.font.name = "Times New Roman"
    r.font.size = Pt(size)
    r.bold = bold
    return p


def add_heading(doc, text, level=1):
    p = doc.add_heading(text, level=level)
    for run in p.runs:
        run.font.name = "Times New Roman"
        run.font.color.rgb = RGBColor(31, 78, 121)
    return p


def add_table(doc, headers, rows, widths=None):
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    hdr = table.rows[0].cells
    for i, h in enumerate(headers):
        set_cell(hdr[i], h, bold=True, color="FFFFFF")
        shade_cell(hdr[i], "1F4E79")
    for row in rows:
        cells = table.add_row().cells
        for i, val in enumerate(row):
            set_cell(cells[i], str(val), size=9.5)
            cells[i].vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    if widths:
        for row in table.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Cm(w)
    doc.add_paragraph()
    return table


def add_code(doc, title, code):
    add_para(doc, title, bold=True, size=11, space_after=3)
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = table.cell(0, 0)
    shade_cell(cell, "F3F6FA")
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
    for n, line in enumerate(code.strip().splitlines()):
        if n:
            p.add_run("\n")
        r = p.add_run(line)
        r.font.name = "Consolas"
        r.font.size = Pt(8)
        r.font.color.rgb = RGBColor(32, 42, 56)
    doc.add_paragraph()


def chapter_page(doc, number, title):
    doc.add_page_break()
    for _ in range(8):
        doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(f"CHAPTER {number}")
    r.font.name = "Times New Roman"
    r.font.size = Pt(22)
    r.bold = True
    r.font.color.rgb = RGBColor(31, 78, 121)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(title.upper())
    r.font.name = "Times New Roman"
    r.font.size = Pt(18)
    r.bold = True
    doc.add_page_break()


def body_paragraphs(doc, paragraphs):
    for text in paragraphs:
        add_para(doc, text, size=11.5, space_after=8)


def build():
    doc = Document()
    sec = doc.sections[0]
    sec.top_margin = Cm(2)
    sec.bottom_margin = Cm(2)
    sec.left_margin = Cm(2.2)
    sec.right_margin = Cm(2.2)

    styles = doc.styles
    styles["Normal"].font.name = "Times New Roman"
    styles["Normal"].font.size = Pt(11.5)
    for style_name, size in [("Heading 1", 16), ("Heading 2", 14), ("Heading 3", 12)]:
        style = styles[style_name]
        style.font.name = "Times New Roman"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor(31, 78, 121)

    # Cover Page
    add_para(doc, "Brain Tumor MRI Classification and Segmentation Using Deep Learning", WD_ALIGN_PARAGRAPH.CENTER, True, 18, 10)
    add_para(doc, "B. Tech Project Report", WD_ALIGN_PARAGRAPH.CENTER, True, 15, 16)
    add_para(doc, "Submitted in partial fulfilment of the requirements for the award of the Degree", WD_ALIGN_PARAGRAPH.CENTER, False, 12, 4)
    add_para(doc, "BACHELOR OF TECHNOLOGY", WD_ALIGN_PARAGRAPH.CENTER, True, 14, 4)
    add_para(doc, "in", WD_ALIGN_PARAGRAPH.CENTER, False, 12, 4)
    add_para(doc, "COMPUTER SCIENCE AND ENGINEERING", WD_ALIGN_PARAGRAPH.CENTER, True, 14, 20)
    add_para(doc, "by", WD_ALIGN_PARAGRAPH.CENTER, False, 12, 8)
    add_para(doc, "[Student Name 1]  [Roll Number]", WD_ALIGN_PARAGRAPH.CENTER, True, 12, 3)
    add_para(doc, "[Student Name 2]  [Roll Number]", WD_ALIGN_PARAGRAPH.CENTER, True, 12, 3)
    add_para(doc, "[Student Name 3]  [Roll Number]", WD_ALIGN_PARAGRAPH.CENTER, True, 12, 18)
    add_para(doc, "Under the Guidance of", WD_ALIGN_PARAGRAPH.CENTER, False, 12, 6)
    add_para(doc, "[Guide Name], [Qualification]", WD_ALIGN_PARAGRAPH.CENTER, True, 12, 22)
    add_para(doc, "DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING", WD_ALIGN_PARAGRAPH.CENTER, True, 13, 3)
    add_para(doc, "[COLLEGE / UNIVERSITY NAME]", WD_ALIGN_PARAGRAPH.CENTER, True, 13, 3)
    add_para(doc, "[CITY, STATE, INDIA]", WD_ALIGN_PARAGRAPH.CENTER, True, 12, 3)
    add_para(doc, "2025-2026", WD_ALIGN_PARAGRAPH.CENTER, True, 12, 0)
    doc.add_page_break()

    # Certificate
    add_para(doc, "CERTIFICATE", WD_ALIGN_PARAGRAPH.CENTER, True, 16, 24)
    body_paragraphs(doc, [
        "This is to certify that the project entitled \"Brain Tumor MRI Classification and Segmentation Using Deep Learning\" is a bona fide record of the project work carried out by the students listed on the cover page under my supervision and guidance.",
        "The project has been submitted in partial fulfilment of the requirements for the award of the Degree of Bachelor of Technology in Computer Science and Engineering. The work presented in this report has not been submitted elsewhere for the award of any degree or diploma.",
    ])
    for _ in range(5):
        doc.add_paragraph()
    add_para(doc, "Project Guide                                         Head / Coordinator", size=11.5)
    add_para(doc, "Department of CSE                                  Department of CSE", size=11.5)
    doc.add_page_break()

    # Declaration
    add_para(doc, "DECLARATION", WD_ALIGN_PARAGRAPH.CENTER, True, 16, 24)
    body_paragraphs(doc, [
        "We hereby declare that the project entitled \"Brain Tumor MRI Classification and Segmentation Using Deep Learning\" has been carried out by us under the guidance of the project supervisor. This report is prepared based on the implementation, experiments, observations, and analysis performed during the project period.",
        "We further declare that this work is original to the best of our knowledge and has not been previously submitted for the award of any degree, diploma, or certificate.",
    ])
    for _ in range(6):
        doc.add_paragraph()
    add_para(doc, "Place: ____________________", size=11.5)
    add_para(doc, "Date:  ____________________", size=11.5)
    add_para(doc, "Student Signatures: ______________________________", size=11.5)
    doc.add_page_break()

    # Acknowledgements
    add_para(doc, "ACKNOWLEDGEMENTS", WD_ALIGN_PARAGRAPH.CENTER, True, 16, 20)
    body_paragraphs(doc, [
        "We express our sincere gratitude to our project guide for valuable guidance, continuous support, and constructive suggestions throughout the development of this project.",
        "We are thankful to the Department of Computer Science and Engineering for providing the academic environment, laboratory facilities, and encouragement needed to complete the work successfully.",
        "We also thank our faculty members, classmates, friends, and family for their support, motivation, and assistance during the development, testing, and documentation of this project.",
    ])
    add_para(doc, "With gratitude,", size=11.5)
    add_para(doc, "Project Team", bold=True, size=11.5)
    doc.add_page_break()

    # Abstract
    add_para(doc, "ABSTRACT", WD_ALIGN_PARAGRAPH.CENTER, True, 16, 20)
    body_paragraphs(doc, [
        "Brain tumors are among the most serious neurological conditions and require accurate diagnosis for timely treatment planning. Magnetic Resonance Imaging (MRI) is widely used for brain tumor diagnosis because it provides high-quality structural information about brain tissues. However, manual interpretation of MRI scans can be time-consuming and depends heavily on expert radiological experience.",
        "This project, BrainTumorAI, proposes a deep learning based system for brain tumor MRI classification and segmentation. The system classifies MRI images into glioma, meningioma, no tumor, and pituitary tumor categories. It uses transfer learning and ensemble learning with convolutional neural network architectures, including EfficientNet, ResNet with CBAM attention, and DenseNet. The system also includes U-Net based segmentation and Grad-CAM based explainability to highlight important tumor regions.",
        "The proposed solution includes dataset preparation, preprocessing, augmentation, model training, final testing, API integration, Gradio-based user interface, visualization, and PDF report generation. The system is designed as an AI-assisted diagnostic support tool, not as a replacement for medical professionals. Experimental results demonstrate the potential of deep learning and explainable AI in medical image analysis.",
        "Keywords: Brain tumor, MRI, deep learning, CNN, transfer learning, ensemble learning, U-Net, Grad-CAM, medical image analysis, computer-aided diagnosis.",
    ])
    doc.add_page_break()

    # Contents and lists
    add_para(doc, "CONTENTS", WD_ALIGN_PARAGRAPH.CENTER, True, 16, 18)
    add_table(doc, ["Chapter No", "Chapter Name", "Page No"], [
        ("1", "Introduction", "1"),
        ("2", "Literature Review", "7"),
        ("3", "Project Description", "12"),
        ("4", "Brain Tumor and MRI Imaging", "19"),
        ("5", "Deep Learning Concepts", "25"),
        ("6", "Model Architecture", "31"),
        ("7", "Methodology", "38"),
        ("8", "Results and Discussion", "50"),
        ("9", "Conclusion and Future Work", "57"),
        ("10", "References", "61"),
        ("11", "Appendices", "64"),
    ], [3, 10, 3])
    doc.add_page_break()

    add_para(doc, "LIST OF FIGURES", WD_ALIGN_PARAGRAPH.CENTER, True, 16, 18)
    add_table(doc, ["Fig No", "Caption", "Chapter"], [
        ("1.1", "Background and Motivation of BrainTumorAI", "1"),
        ("3.1", "Proposed System Workflow", "3"),
        ("6.1", "Ensemble Classification Architecture", "6"),
        ("7.1", "Training and Evaluation Pipeline", "7"),
        ("8.1", "Performance Evaluation Metrics", "8"),
    ], [3, 10, 3])
    add_para(doc, "LIST OF ACRONYMS", WD_ALIGN_PARAGRAPH.CENTER, True, 16, 18)
    add_table(doc, ["Acronym", "Full Form"], [
        ("AI", "Artificial Intelligence"),
        ("API", "Application Programming Interface"),
        ("AUC", "Area Under the Curve"),
        ("CBAM", "Convolutional Block Attention Module"),
        ("CNN", "Convolutional Neural Network"),
        ("DL", "Deep Learning"),
        ("MRI", "Magnetic Resonance Imaging"),
        ("TTA", "Test Time Augmentation"),
        ("U-Net", "U-shaped Convolutional Network for Segmentation"),
        ("XAI", "Explainable Artificial Intelligence"),
    ], [4, 11])

    # Chapter 1
    chapter_page(doc, 1, "Introduction")
    add_heading(doc, "1.1 Background and Motivation", 1)
    body_paragraphs(doc, [
        "Brain tumor diagnosis is a critical area in medical imaging because the presence, type, and location of a tumor directly influence clinical decision-making. MRI scans provide detailed anatomical information and are widely used for detecting abnormal brain tissue. Nevertheless, manual interpretation requires expertise, time, and careful analysis.",
        "The motivation for this project is to design an AI-assisted diagnostic support system that can classify MRI images and provide visual explanations. The system aims to reduce diagnostic workload, improve consistency, and support radiologists with interpretable predictions.",
    ])
    if BACKGROUND_IMG.exists():
        doc.add_picture(str(BACKGROUND_IMG), width=Inches(6.3))
        add_para(doc, "Fig 1.1: Background and Motivation of BrainTumorAI", WD_ALIGN_PARAGRAPH.CENTER, False, 10, 10)
    add_heading(doc, "1.2 Aim of the Project", 1)
    body_paragraphs(doc, [
        "The main aim of the project is to develop a deep learning system that classifies brain MRI images into glioma, meningioma, no tumor, and pituitary tumor categories. The project also aims to generate segmentation masks and explainability heatmaps for better interpretation of model decisions.",
    ])
    add_heading(doc, "1.3 Project Domain", 1)
    body_paragraphs(doc, [
        "The project belongs to the domains of artificial intelligence, computer vision, medical image processing, and healthcare decision support. It applies convolutional neural networks and explainable AI methods to the analysis of brain MRI images.",
    ])
    add_heading(doc, "1.4 Problem Statement", 1)
    body_paragraphs(doc, [
        "Given a brain MRI image, the system must automatically identify the tumor class, estimate prediction confidence, generate class probabilities, and optionally highlight suspicious regions through Grad-CAM and segmentation outputs.",
    ])

    # Chapter 2
    chapter_page(doc, 2, "Literature Review")
    add_heading(doc, "2.1 Overview of Existing Research", 1)
    body_paragraphs(doc, [
        "Traditional medical image classification relied on handcrafted features such as texture descriptors, intensity histograms, edge features, and classical machine learning models. Although these methods are useful, their performance depends strongly on feature engineering and dataset quality.",
        "Deep learning has improved medical image analysis because CNNs can learn hierarchical features directly from images. Transfer learning further improves performance by using pretrained models as feature extractors and fine-tuning them for medical datasets.",
    ])
    add_heading(doc, "2.2 Relevance of CNNs in Medical Imaging", 1)
    body_paragraphs(doc, [
        "CNN architectures such as ResNet, EfficientNet, DenseNet, and Vision Transformers have been successfully applied to classification tasks. Their ability to learn spatial patterns makes them suitable for MRI analysis. Attention mechanisms such as CBAM help the network focus on important channels and spatial regions.",
    ])
    add_heading(doc, "2.3 Research Gap", 1)
    body_paragraphs(doc, [
        "Many systems focus only on classification accuracy. However, medical applications also require interpretability and localization. BrainTumorAI addresses this by combining ensemble classification, U-Net segmentation, Grad-CAM visualization, and report generation.",
    ])

    # Chapter 3
    chapter_page(doc, 3, "Project Description")
    add_heading(doc, "3.1 Existing System", 1)
    body_paragraphs(doc, [
        "Existing diagnosis depends primarily on manual MRI interpretation by radiologists. Classical automated systems use handcrafted features and machine learning classifiers. These systems may struggle with variations in image quality, tumor size, tumor shape, and scan orientation.",
    ])
    add_heading(doc, "3.2 Proposed System", 1)
    body_paragraphs(doc, [
        "The proposed BrainTumorAI system uses a deep learning ensemble to improve classification robustness. It accepts MRI images, preprocesses them, applies test-time augmentation, predicts tumor category, generates Grad-CAM heatmaps, produces segmentation masks, and prepares a diagnostic report.",
    ])
    add_table(doc, ["Stage", "Description"], [
        ("Input", "Brain MRI image uploaded through UI or API"),
        ("Preprocessing", "Resize, normalize, and convert image to tensor"),
        ("Classification", "EfficientNet, ResNet-CBAM, and DenseNet ensemble"),
        ("Localization", "Grad-CAM heatmap and U-Net segmentation"),
        ("Output", "Prediction, confidence, probabilities, visual explanation, report"),
    ], [4, 11])
    add_heading(doc, "3.3 System Specifications", 1)
    add_table(doc, ["Requirement Type", "Specification"], [
        ("Hardware", "CPU supported; GPU recommended for training and faster inference"),
        ("Language", "Python"),
        ("Frameworks", "PyTorch, torchvision, timm, OpenCV, Albumentations"),
        ("Interface", "Gradio web interface and FastAPI backend"),
        ("Output", "Prediction results, heatmaps, segmentation masks, PDF reports"),
    ], [4, 11])

    # Chapter 4
    chapter_page(doc, 4, "Brain Tumor and MRI Imaging")
    add_heading(doc, "4.1 Brain Tumor", 1)
    body_paragraphs(doc, [
        "A brain tumor is an abnormal growth of cells within or around the brain. Tumors may be benign or malignant, and their symptoms depend on location, size, and growth rate. Common tumor categories considered in this project include glioma, meningioma, and pituitary tumor, along with no tumor cases.",
    ])
    add_heading(doc, "4.2 MRI Imaging", 1)
    body_paragraphs(doc, [
        "Magnetic Resonance Imaging is a non-invasive imaging method that provides detailed visualization of soft tissues. MRI is highly suitable for brain tumor detection because it can reveal structural abnormalities and tissue contrast.",
    ])
    add_heading(doc, "4.3 Importance of Automated Support", 1)
    body_paragraphs(doc, [
        "Automated diagnostic support can help screen images, reduce workload, and provide consistent second-opinion results. In this project, AI predictions are combined with visual explanations so that users can understand the model's focus regions.",
    ])

    # Chapter 5
    chapter_page(doc, 5, "Deep Learning Concepts")
    add_heading(doc, "5.1 Deep Learning", 1)
    body_paragraphs(doc, [
        "Deep learning is a branch of machine learning that uses neural networks with multiple layers to learn representations from data. In image classification, convolutional layers extract local patterns such as edges, textures, shapes, and high-level structures.",
    ])
    add_heading(doc, "5.2 Transfer Learning", 1)
    body_paragraphs(doc, [
        "Transfer learning uses pretrained networks trained on large image datasets and adapts them to a target domain. This is useful in medical imaging where labelled data may be limited. BrainTumorAI uses pretrained CNN backbones and fine-tunes them for MRI classification.",
    ])
    add_heading(doc, "5.3 Explainable AI", 1)
    body_paragraphs(doc, [
        "Explainable AI techniques help users interpret model predictions. Grad-CAM generates a heatmap showing image regions that strongly influenced the model decision. This improves transparency and supports trust in AI-assisted diagnosis.",
    ])

    # Chapter 6
    chapter_page(doc, 6, "Model Architecture")
    add_heading(doc, "6.1 Classification Models", 1)
    body_paragraphs(doc, [
        "The project uses an ensemble of CNN-based classifiers. EfficientNet extracts efficient multi-scale features, ResNet-CBAM applies residual learning with attention, and DenseNet improves feature reuse through dense connections.",
    ])
    add_heading(doc, "6.2 Ensemble Learning", 1)
    body_paragraphs(doc, [
        "Ensemble learning combines predictions from multiple models. In this project, soft-voting is applied using model weights. The final probability distribution is obtained by combining calibrated probabilities from individual models.",
    ])
    add_heading(doc, "6.3 Segmentation Architecture", 1)
    body_paragraphs(doc, [
        "The segmentation module uses U-Net, which consists of encoder and decoder paths connected with skip connections. This architecture is suitable for medical image segmentation because it preserves spatial details while learning high-level context.",
    ])

    # Chapter 7
    chapter_page(doc, 7, "Methodology")
    add_heading(doc, "7.1 Data Acquisition and Preparation", 1)
    body_paragraphs(doc, [
        "MRI images are arranged in class-wise folders for training and testing. The data preparation script scans folders, assigns labels, creates metadata CSV files, and performs stratified train-validation splitting.",
    ])
    add_heading(doc, "7.2 Preprocessing and Augmentation", 1)
    body_paragraphs(doc, [
        "Each image is resized to 224 x 224 pixels, converted to RGB format, normalized using ImageNet statistics, and converted to tensor format. Augmentation techniques such as rotation, flipping, color jitter, MixUp, and CutMix improve model generalization.",
    ])
    add_heading(doc, "7.3 Training Process", 1)
    body_paragraphs(doc, [
        "Training is performed in phases. Initially, the backbone may be frozen while the classification head learns. Later, the complete model is fine-tuned using AdamW optimizer and cosine learning rate scheduling. The best model is saved based on validation AUC.",
    ])
    add_heading(doc, "7.4 Inference Process", 1)
    body_paragraphs(doc, [
        "During inference, uploaded images are preprocessed and passed through the ensemble. Test-time augmentation is used to average predictions from multiple transformed views. The system returns class name, confidence, probabilities, heatmap, segmentation mask, and latency information.",
    ])

    # Chapter 8
    chapter_page(doc, 8, "Results and Discussion")
    add_heading(doc, "8.1 Evaluation Metrics", 1)
    body_paragraphs(doc, [
        "The model is evaluated using accuracy, AUC, sensitivity, specificity, F1-score, and confusion matrix. These metrics are suitable for medical classification because they measure both overall correctness and class-wise diagnostic performance.",
    ])
    add_table(doc, ["Metric", "Purpose"], [
        ("Accuracy", "Measures overall correct predictions"),
        ("AUC", "Measures discrimination ability across classes"),
        ("Sensitivity", "Measures ability to detect positive class cases"),
        ("Specificity", "Measures ability to avoid false positives"),
        ("F1-score", "Balances precision and recall"),
        ("Confusion Matrix", "Shows class-wise correct and incorrect predictions"),
    ], [4, 11])
    add_heading(doc, "8.2 Observations", 1)
    body_paragraphs(doc, [
        "The ensemble approach improves robustness by combining complementary strengths of different CNN architectures. Explainability outputs help verify whether the model focuses on medically meaningful image regions. Segmentation results provide additional visual evidence for tumor localization.",
    ])
    add_heading(doc, "8.3 Discussion", 1)
    body_paragraphs(doc, [
        "The system demonstrates the usefulness of AI in medical image analysis. However, it should be used as a decision-support tool and not as a final clinical diagnosis system. Proper validation with diverse clinical datasets is necessary before real-world deployment.",
    ])

    # Chapter 9
    chapter_page(doc, 9, "Conclusion and Future Work")
    add_heading(doc, "9.1 Conclusion", 1)
    body_paragraphs(doc, [
        "BrainTumorAI successfully integrates MRI image preprocessing, deep learning classification, ensemble prediction, explainability, segmentation, API deployment, and a user-friendly web interface. The project demonstrates how AI can support medical image diagnosis by producing predictions and visual explanations.",
    ])
    add_heading(doc, "9.2 Future Enhancements", 1)
    body_paragraphs(doc, [
        "Future work can include larger multi-hospital datasets, DICOM support, improved segmentation using annotated masks, cloud deployment, role-based login, database-backed patient history, automated report storage, and clinical validation with expert radiologists.",
    ])

    # Chapter 10
    chapter_page(doc, 10, "References")
    refs = [
        "Amin et al., A distinctive approach in brain tumor detection and classification using MRI, Pattern Recognition Letters, 2020.",
        "He et al., Deep Residual Learning for Image Recognition, IEEE CVPR, 2016.",
        "Ronneberger et al., U-Net: Convolutional Networks for Biomedical Image Segmentation, MICCAI, 2015.",
        "Selvaraju et al., Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization, ICCV, 2017.",
        "Tan and Le, EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks, ICML, 2019.",
        "Huang et al., Densely Connected Convolutional Networks, IEEE CVPR, 2017.",
    ]
    for i, ref in enumerate(refs, 1):
        add_para(doc, f"[{i}] {ref}", size=11.5)

    # Chapter 11
    chapter_page(doc, 11, "Appendices")
    add_heading(doc, "Appendix A: Important Code Snippets", 1)
    add_code(doc, "Dataset Loading", """
class MRIDataset(Dataset):
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = cv2.imread(str(row["path"]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)["image"]
        return image, int(row["label"])
""")
    add_code(doc, "Ensemble Prediction", """
class Ensemble(nn.Module):
    def forward(self, x):
        output = None
        for name, model in self.models.items():
            logits = model(x) / self.temperatures[name].clamp(min=0.1)
            probs = torch.softmax(logits, dim=-1) * self.weights[name]
            output = probs if output is None else output + probs
        return output
""")
    add_code(doc, "API Endpoint", """
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    result = _engine.predict(np.array(image))
    return JSONResponse(content=result)
""")
    add_heading(doc, "Appendix B: Project Folder Highlights", 1)
    add_table(doc, ["Folder / File", "Description"], [
        ("app.py", "Main Gradio application for diagnosis and report generation"),
        ("train.py", "Training script for classification models and ensemble"),
        ("prepare_data.py", "Dataset scanning and CSV preparation"),
        ("final_test.py", "Final holdout testing and metrics generation"),
        ("src/api", "FastAPI backend and inference pipeline"),
        ("segmentation", "U-Net and post-processing logic"),
        ("inference", "Grad-CAM and segmentation inference helpers"),
        ("report", "PDF report generation modules"),
    ], [5, 10])

    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    build()
