from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


OUTPUT_PATH = "BrainTumorAI_Project_Review_Answered_Questions.pdf"


PROJECT_FACTS = [
    ["Project", "BrainTumorAI / NeuroScan AI"],
    ["Task", "Brain MRI classification and assistive diagnostic report generation"],
    ["Classes", "Glioma, Meningioma, No Tumor, Pituitary"],
    ["Input size", "224 x 224 MRI images"],
    ["Main models", "EfficientNet-B3, ResNet50 + CBAM, DenseNet121"],
    ["Ensemble", "Soft-voting weighted ensemble: 40%, 30%, 30%"],
    ["Inference", "6-view test-time augmentation and averaged probabilities"],
    ["Explainability", "Grad-CAM heatmap support"],
    ["Segmentation", "Attention U-Net / U-Net style tumor mask support"],
    ["Interface", "Gradio web app with PDF report generation"],
    ["Reported dashboard metrics", "AUC 0.9996, Accuracy 98.78%, Sensitivity 98.68%, Specificity 99.59%"],
    ["Earlier CSV summary", "Ensemble+TTA accuracy 99.31%, AUC 0.9968"],
]


sections = [
    (
        "1. Problem Statement & Motivation",
        "Our project addresses the challenge of supporting brain tumor identification from MRI scans. Manual review by specialists is essential, but it can be time-consuming and depends on image quality, expert availability, and workload. The motivation of BrainTumorAI is to build an assistive AI system that can classify MRI images, show confidence values, provide visual explanation, and generate a patient-oriented report for review.",
        "Strong review answer: The problem is not simply 'detecting tumors'; it is building a reliable decision-support prototype for classifying MRI images into glioma, meningioma, pituitary tumor, or no tumor. The system is meant to support screening and educational analysis, not replace radiologists.",
        "Model answer: BrainTumorAI solves the problem of classifying brain MRI images into tumor categories and no-tumor cases. This is important because early, consistent, and explainable image analysis can support clinical review and reduce diagnostic delay. Our system provides classification, confidence, heatmap-based explanation, segmentation support, and a downloadable report.",
    ),
    (
        "2. Literature Review & Research Gap",
        "Earlier work has used classical machine learning methods such as SVM with handcrafted features like GLCM, as well as modern CNN-based deep learning. The baseline referenced in the project is Amin et al. 2020 using SVM+GLCM, with around 97.1% accuracy and AUC 0.980. Our gap is that classical methods depend heavily on manual feature extraction, while many deep learning systems focus only on classification and do not provide a usable report or explanation layer.",
        "Strong review answer: We reviewed both classical feature-based methods and deep learning approaches. Our project improves the practical pipeline by combining ensemble learning, TTA, Grad-CAM explanation, segmentation support, and PDF report generation.",
        "Model answer: Existing systems show that MRI tumor classification is feasible, but classical methods often depend on handcrafted features and many AI demos stop at prediction only. Our project addresses this gap by building an end-to-end system that classifies MRI scans, explains predictions using Grad-CAM, supports tumor localization, and generates a structured PDF report.",
    ),
    (
        "3. Research Objectives & Hypotheses",
        "The main objective is to develop a deep-learning-based system for classifying MRI brain scans into four classes. Secondary objectives include evaluating performance using accuracy, AUC, sensitivity, specificity, and F1-style metrics; adding explainability through Grad-CAM; supporting segmentation; and generating a patient report.",
        "Strong review answer: Objectives should be measurable. For this project, the measurable goals are four-class classification, reliable test performance, visual explanation, and report generation.",
        "Model answer: The main objective is to classify MRI images into glioma, meningioma, pituitary tumor, or no tumor using an ensemble model. The hypothesis is that combining multiple CNN models with test-time augmentation will produce more robust predictions than a single model or classical feature extraction. We also aimed to make the result interpretable through heatmaps and usable through a Gradio interface and PDF report.",
    ),
    (
        "4. Methodology & Research Design",
        "The methodology follows an experimental system-development design. MRI images are loaded, resized to 224 x 224, normalized, augmented during training, passed through deep learning classifiers, combined using ensemble voting, and evaluated on test data. The application layer then presents prediction, confidence, probabilities, segmentation/heatmap outputs, and a PDF report.",
        "Strong review answer: Explain the full pipeline: data preparation, model training, ensemble inference, evaluation, explainability, and deployment interface.",
        "Model answer: Our methodology begins with MRI preprocessing and train-validation-test organization. We trained multiple deep learning models and combined them in a soft-voting ensemble with test-time augmentation. Finally, we evaluated classification performance and integrated the model into a Gradio application with visual explanations and report generation.",
    ),
    (
        "5. Data Collection & Sampling Strategy",
        "The configuration points to a local dataset under C:/Users/HP/Desktop/MainProject/Project with Training and Testing folders. The project uses four classes: glioma, meningioma, notumor, and pituitary. The training configuration uses a 20% validation split from the training folder and keeps the testing folder separate for final evaluation.",
        "Strong review answer: Mention the data source, class labels, split strategy, and why test data must remain unseen.",
        "Model answer: The dataset is organized into Training and Testing folders with four MRI classes: glioma, meningioma, no tumor, and pituitary. From the training set, 20% was used for validation, while the separate test set was used to evaluate final performance. This separation helps ensure that the model is tested on unseen images rather than memorized training examples.",
    ),
    (
        "6. Tools, Technologies & Frameworks Used",
        "The project uses Python as the main language. PyTorch and timm are used for deep learning models, OpenCV/PIL/NumPy for image processing, Gradio for the user interface, ReportLab for PDF report generation, and FastAPI support exists for API-style deployment. Checkpoints are stored locally and the app runs in CPU mode in the current configuration.",
        "Strong review answer: Do not just list tools. Explain the role of each tool in the pipeline.",
        "Model answer: Python was chosen because it has strong support for machine learning and medical image processing. PyTorch and timm were used to build and load EfficientNet, ResNet-CBAM, and DenseNet models; OpenCV and PIL handled preprocessing; Gradio provided the user interface; and ReportLab generated patient PDF reports.",
    ),
    (
        "7. Results & Findings",
        "The app/review dashboard reports strong final performance: AUC 0.9996, accuracy 98.78%, sensitivity 98.68%, and specificity 99.59% on a test set display of 1,311 images. Earlier CSV summaries list ensemble+TTA accuracy as 99.31% and AUC as 0.9968. The important finding is that ensemble prediction improves robustness compared with individual models.",
        "Strong review answer: Give the actual metrics and explain what they mean.",
        "Model answer: The system achieved high classification performance, with the dashboard reporting AUC 0.9996, accuracy 98.78%, sensitivity 98.68%, and specificity 99.59%. The results suggest that the ensemble learned useful MRI patterns and that TTA helped stabilize predictions. However, these results should still be interpreted as project-level validation, not direct clinical approval.",
    ),
    (
        "8. Analysis & Interpretation",
        "The ensemble works well because the models learn complementary visual patterns. EfficientNet captures scaled CNN features, ResNet50+CBAM adds channel and spatial attention, and DenseNet reuses dense feature connections. TTA averages predictions across transformed views, reducing sensitivity to orientation and minor image variation.",
        "Strong review answer: Interpret why the method performed well and where errors may still happen.",
        "Model answer: The results indicate that the model can identify class-specific MRI patterns, especially when predictions from multiple architectures are combined. Some errors may occur when tumor boundaries are unclear, images are low quality, or different tumor types have visually similar regions. That is why the system provides confidence scores and should be used as assistive support rather than an independent diagnosis.",
    ),
    (
        "9. Limitations of the Study",
        "The main limitations are dataset dependence, possible class imbalance, limited real-hospital validation, and sensitivity to image acquisition differences. The segmentation component may depend on pseudo-mask or limited mask quality, and Grad-CAM provides approximate explanation rather than exact medical reasoning. The system is a prototype and not a regulated clinical device.",
        "Strong review answer: Be honest, but do not weaken the project unnecessarily. State what the limitation means and how future work can address it.",
        "Model answer: A key limitation is that the model was trained and tested on available dataset folders, so its performance may not generalize to every hospital scanner or patient population. The segmentation and heatmap outputs are helpful for interpretation but should not be treated as final clinical evidence. The project is best presented as an assistive AI prototype requiring further validation.",
    ),
    (
        "10. Contributions & Novelty",
        "The contribution is an integrated BrainTumorAI system rather than only a standalone model. It combines ensemble MRI classification, TTA, attention-based architecture, Grad-CAM explanation, segmentation support, and PDF report generation in a usable Gradio application.",
        "Strong review answer: Novelty can be integration, improvement, usability, or evaluation, not only inventing a new algorithm.",
        "Model answer: The contribution of this project is an end-to-end brain tumor AI prototype that goes beyond a simple classifier. It integrates multiple deep learning models, confidence-based prediction, visual explanation, segmentation support, and downloadable report generation. This makes the system more practical for project demonstration and decision-support research.",
    ),
    (
        "11. Future Work & Scope",
        "Future work should focus on external validation using hospital-grade datasets, stronger segmentation labels, better explainability, calibration testing, and deployment improvements. Additional work could include DICOM support, radiologist feedback, patient-level splitting, uncertainty estimation, and regulatory/clinical validation planning.",
        "Strong review answer: Future work should directly address current limitations.",
        "Model answer: In future, we can train and validate the system on larger multi-center MRI datasets and include stronger expert-labeled segmentation masks. We can also add DICOM support, uncertainty estimation, and radiologist feedback to improve real-world usefulness. With proper validation, the project could evolve into a clinical decision-support tool.",
    ),
    (
        "12. Ethical Considerations",
        "Because the system deals with medical images, privacy, consent, bias, and responsible use are important. The system should not claim to replace doctors. If real patient data is used, it must be anonymized and handled securely.",
        "Strong review answer: Mention privacy, human oversight, data bias, and risk of over-reliance.",
        "Model answer: Ethical use is important because incorrect medical AI predictions can affect patient care. Our system should be used only as an assistive tool, with final decisions made by qualified medical professionals. Patient data must be anonymized, stored securely, and evaluated for bias across different populations and scanners.",
    ),
    (
        "13. Practical / Real-world Applications",
        "The project can be used for educational demonstrations, research prototypes, AI-assisted screening studies, and preliminary decision-support workflows. The PDF report feature makes it useful for presenting prediction, confidence, image, and clinical notes in a readable format.",
        "Strong review answer: Be realistic. Say where it can help now and what is needed before hospital use.",
        "Model answer: Practically, BrainTumorAI can help students, researchers, and clinicians understand how AI can support MRI classification. The Gradio interface and PDF report make the result easy to review and share. For real hospital use, the model would need external validation, regulatory approval, and expert supervision.",
    ),
    (
        "14. Defense of Design Choices",
        "CNN-based models were chosen because MRI classification is an image-recognition task where spatial features matter. An ensemble was chosen because different architectures make different errors, and soft-voting improves robustness. TTA was added to reduce prediction instability caused by image orientation and small transformations. Gradio was selected for fast, usable demonstration; ReportLab was used for structured PDF generation.",
        "Strong review answer: Compare alternatives and name trade-offs. CNNs need more data and computation, but they avoid manual feature design.",
        "Model answer: We used a deep learning ensemble because CNN models can automatically learn MRI image features better than manual feature extraction methods such as GLCM. EfficientNet, ResNet-CBAM, and DenseNet were combined because each architecture captures features differently, and ensemble voting reduces dependence on one model. The trade-off is higher computational cost, but the benefit is stronger and more stable prediction.",
    ),
]


question_answers = [
    ("Basic", "What is the main problem your project addresses?", "It addresses brain MRI classification into glioma, meningioma, pituitary tumor, and no tumor. The aim is to provide an assistive AI system with confidence values, visual explanation, and a report, not to replace doctors."),
    ("Basic", "Why did you choose this topic?", "Brain tumor detection is a meaningful healthcare problem where timely image interpretation matters. It is also technically suitable for deep learning because MRI images contain visual patterns that CNNs can learn."),
    ("Basic", "What are the main objectives of your project?", "The main objectives are four-class MRI classification, reliable evaluation using medical-style metrics, Grad-CAM-based explanation, segmentation support, and PDF report generation through a usable interface."),
    ("Basic", "Who are the intended users or beneficiaries?", "The immediate users are students, researchers, and reviewers evaluating an AI medical imaging prototype. In future, with validation, radiologists or clinicians could use similar systems as decision-support tools."),
    ("Basic", "What tools and technologies did you use?", "The project uses Python, PyTorch, timm, OpenCV/PIL, NumPy, Gradio, ReportLab, and supporting modules for segmentation and Grad-CAM."),
    ("Basic", "What dataset or data source did you use?", "The project uses a local MRI dataset organized into Training and Testing folders with four classes: glioma, meningioma, notumor, and pituitary. The configuration keeps a separate test folder and uses a validation split from training data."),
    ("Basic", "What are the main features of your system?", "The system supports MRI upload, class prediction, confidence score, all-class probabilities, Grad-CAM/visual explanation, segmentation support, risk-level notes, and downloadable PDF report generation."),
    ("Intermediate", "What gap did you identify in existing work?", "Many earlier systems either use handcrafted features or focus only on prediction accuracy. Our project integrates prediction, explainability, segmentation support, and report generation in one workflow."),
    ("Intermediate", "How is your project different from previous studies?", "It uses a three-model ensemble with TTA and a practical Gradio interface. It also adds report generation and visual explanation, making the output more understandable for review."),
    ("Intermediate", "Why did you choose this methodology?", "An experimental deep learning methodology fits the problem because MRI classification can be evaluated quantitatively using unseen test images. The system-development part is needed because the project also includes a usable application and report downloader."),
    ("Intermediate", "How did you preprocess or prepare your data?", "Images are resized to 224 x 224, normalized, and augmented during training using transformations such as flips, rotations, noise/blur, and regularization methods like MixUp or CutMix in the configured pipeline."),
    ("Intermediate", "How did you evaluate your results?", "The system is evaluated using accuracy, AUC, sensitivity, specificity, F1/precision-style metrics, confusion matrix, ROC/PR curves, and visual inspection of heatmaps and segmentation outputs."),
    ("Intermediate", "Why did you use these performance metrics?", "Accuracy gives overall correctness, AUC measures ranking ability across thresholds, sensitivity checks how well tumor cases are detected, and specificity checks how well non-tumor or negative decisions are controlled."),
    ("Intermediate", "What were the major findings of your project?", "The major finding is that ensemble-based classification with TTA gives strong performance and stable predictions. The dashboard reports AUC 0.9996 and accuracy 98.78%, with high sensitivity and specificity."),
    ("Intermediate", "What limitations did you observe?", "The project depends on the available dataset and may not generalize to all hospitals or scanners. It also requires stronger clinical validation and should not be used as a standalone diagnosis system."),
    ("Advanced", "Why did you choose this model or approach instead of alternatives?", "CNN-based deep learning avoids manual feature extraction and is strong for image recognition. The ensemble was chosen because EfficientNet, ResNet-CBAM, and DenseNet learn complementary features, while TTA improves robustness."),
    ("Advanced", "How do you know your results are reliable?", "Reliability is supported by separated testing, multiple metrics, and comparison with earlier individual-model results. However, true clinical reliability would require external validation on independent hospital datasets."),
    ("Advanced", "What would happen if your dataset were larger, smaller, or imbalanced?", "A larger balanced dataset would likely improve generalization. A smaller or imbalanced dataset could cause overfitting or bias toward majority classes, so augmentation, weighted sampling, and careful validation are important."),
    ("Advanced", "How would your system perform in a real-world environment?", "It may perform well on images similar to the training/test dataset, but real-world MRI scans can differ by scanner, protocol, quality, and patient population. Real deployment requires validation across multiple sources."),
    ("Advanced", "What ethical risks exist, and how would you reduce them?", "The main risks are privacy leakage, biased predictions, and over-reliance on AI output. These can be reduced through anonymization, secure storage, bias testing, clear disclaimers, and mandatory expert review."),
]


def make_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="TitleCustom",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=22,
        leading=28,
        textColor=colors.HexColor("#17365d"),
        spaceAfter=12,
    ))
    styles.add(ParagraphStyle(
        name="Sub",
        parent=styles["BodyText"],
        alignment=TA_CENTER,
        fontSize=10.5,
        leading=15,
        textColor=colors.HexColor("#4b5563"),
        spaceAfter=18,
    ))
    styles.add(ParagraphStyle(
        name="H",
        parent=styles["Heading2"],
        fontSize=13.5,
        leading=17,
        textColor=colors.HexColor("#1f4e79"),
        spaceBefore=10,
        spaceAfter=5,
        keepWithNext=True,
    ))
    styles.add(ParagraphStyle(
        name="BodyCustom",
        parent=styles["BodyText"],
        fontSize=9.4,
        leading=13.2,
        textColor=colors.HexColor("#263238"),
        spaceAfter=5,
    ))
    styles.add(ParagraphStyle(
        name="Label",
        parent=styles["BodyText"],
        fontSize=9.5,
        leading=12.5,
        textColor=colors.HexColor("#111827"),
        fontName="Helvetica-Bold",
        spaceBefore=4,
        spaceAfter=2,
    ))
    styles.add(ParagraphStyle(
        name="AnswerBox",
        parent=styles["BodyText"],
        fontSize=9.2,
        leading=13,
        textColor=colors.HexColor("#253044"),
        borderColor=colors.HexColor("#bfd0e4"),
        borderWidth=0.6,
        borderPadding=7,
        backColor=colors.HexColor("#f6f9fd"),
        spaceBefore=2,
        spaceAfter=8,
    ))
    return styles


def bullet_list(items, styles):
    return ListFlowable(
        [ListItem(Paragraph(item, styles["BodyCustom"]), leftIndent=10) for item in items],
        bulletType="bullet",
        leftIndent=16,
        bulletFontSize=6,
        bulletColor=colors.HexColor("#1f4e79"),
    )


def header_footer(canvas, doc):
    canvas.saveState()
    width, height = A4
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#6b7280"))
    canvas.drawString(doc.leftMargin, 0.35 * inch, "BrainTumorAI Project Review Answers")
    canvas.drawRightString(width - doc.rightMargin, 0.35 * inch, f"Page {doc.page}")
    canvas.restoreState()


def build_pdf():
    styles = make_styles()
    doc = SimpleDocTemplate(
        OUTPUT_PATH,
        pagesize=A4,
        leftMargin=0.62 * inch,
        rightMargin=0.62 * inch,
        topMargin=0.62 * inch,
        bottomMargin=0.62 * inch,
        title="BrainTumorAI Project Review Answer Bank",
        author="Codex",
    )
    story = []

    story.append(Paragraph("BrainTumorAI Project Review Answer Bank", styles["TitleCustom"]))
    story.append(Paragraph(
        "Project-specific answers for thesis/project viva and review questions. Use these as model answers and adjust the wording to match what you personally implemented.",
        styles["Sub"],
    ))

    story.append(Paragraph("Quick Project Snapshot", styles["H"]))
    table = Table(PROJECT_FACTS, colWidths=[1.7 * inch, 4.9 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eaf2fb")),
        ("BACKGROUND", (1, 0), (1, -1), colors.HexColor("#f8fbff")),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#1f2937")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("LEADING", (0, 0), (-1, -1), 11),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(table)
    story.append(Spacer(1, 0.12 * inch))

    for title, project_answer, strong, model in sections:
        story.append(Paragraph(title, styles["H"]))
        story.append(Paragraph("<b>Project-specific answer:</b>", styles["Label"]))
        story.append(Paragraph(project_answer, styles["BodyCustom"]))
        story.append(Paragraph("<b>What to emphasize:</b>", styles["Label"]))
        story.append(Paragraph(strong, styles["BodyCustom"]))
        story.append(Paragraph("<b>2-3 sentence viva answer:</b>", styles["Label"]))
        story.append(Paragraph(model, styles["AnswerBox"]))

    story.append(PageBreak())
    story.append(Paragraph("Reviewer Question Bank With Project Answers", styles["H"]))
    story.append(Paragraph(
        "These are concise answers you can rehearse. Start with the direct answer, then add one project-specific detail.",
        styles["BodyCustom"],
    ))

    current_level = None
    for level, question, answer in question_answers:
        if level != current_level:
            current_level = level
            story.append(Paragraph(level, styles["H"]))
        story.append(Paragraph(f"<b>Q: {question}</b>", styles["Label"]))
        story.append(Paragraph(f"A: {answer}", styles["AnswerBox"]))

    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph("Final Rehearsal Note", styles["H"]))
    story.append(Paragraph(
        "In review, do not try to sound like a textbook. Sound like the person who built the system: explain what problem you solved, why this design was chosen, what the results show, what the limits are, and how you would improve it next.",
        styles["AnswerBox"],
    ))

    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)


if __name__ == "__main__":
    build_pdf()
    print(OUTPUT_PATH)
