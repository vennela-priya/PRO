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
)


OUTPUT_PATH = "Thesis_Review_Preparation_Guide_BrainTumorAI.pdf"


sections = [
    {
        "title": "1. Problem Statement & Motivation",
        "why": "Reviewers ask this to check whether you understand the real problem, not just the project title. A clear problem statement proves that your work is necessary, focused, and meaningful.",
        "questions": [
            "What exact problem are you solving?",
            "Who is affected by this problem?",
            "Why is this problem important now?",
            "What happens if the problem remains unsolved?",
            "Is the problem practical, research-based, or both?",
        ],
        "strong": "A strong answer clearly defines the problem, explains its importance, and connects it to a real-world or academic need. It avoids vague claims like 'this is useful' and instead explains why the problem matters.",
        "mistakes": [
            "Giving a broad topic instead of a specific problem.",
            "Saying the topic is interesting without explaining the deeper motivation.",
            "Overclaiming the seriousness of the problem.",
            "Not connecting the problem to users, society, industry, or research.",
        ],
        "example": "My project addresses the difficulty of accurately identifying brain tumor conditions from medical images in a timely manner. This is important because early and reliable detection can support clinical decision-making and reduce diagnostic delays. The motivation is to explore how automated image-based analysis can assist, not replace, medical experts.",
    },
    {
        "title": "2. Literature Review & Research Gap",
        "why": "Reviewers ask this to check whether your work is based on existing knowledge and whether you know what has already been done. The research gap shows why your project deserves to exist.",
        "questions": [
            "What previous studies or systems did you review?",
            "What methods have others used?",
            "What are the strengths and weaknesses of existing work?",
            "What gap did you identify?",
            "How does your project address that gap?",
        ],
        "strong": "A good answer summarizes major existing approaches, identifies a clear limitation, and explains how your work responds to that limitation.",
        "mistakes": [
            "Listing papers without comparing them.",
            "Saying no one has done this before without evidence.",
            "Confusing a limitation with a research gap.",
            "Not linking the literature review to your own objectives.",
        ],
        "example": "Previous studies have used machine learning and deep learning methods for brain tumor classification, with CNN-based models showing strong performance on image data. However, many works focus mainly on accuracy and give less attention to usability, preprocessing, or model interpretability. My project tries to address this by combining image preprocessing, classification, and clear output presentation.",
    },
    {
        "title": "3. Research Objectives & Hypotheses",
        "why": "Reviewers ask this to know whether your project has a clear direction. Objectives show what you planned to achieve, while hypotheses show what you expected to test or demonstrate.",
        "questions": [
            "What are the main objectives?",
            "Are the objectives measurable?",
            "What hypothesis are you testing?",
            "How do your objectives relate to the problem?",
            "Did your results meet the objectives?",
        ],
        "strong": "A strong answer gives specific, measurable objectives and connects them directly to methodology and results.",
        "mistakes": [
            "Using vague objectives such as 'to study' or 'to understand'.",
            "Having too many objectives.",
            "Presenting objectives that are not actually tested.",
            "Not knowing the difference between an objective and a hypothesis.",
        ],
        "example": "The main objective is to develop a system that can classify brain MRI images into relevant tumor categories using a trained model. A secondary objective is to evaluate the model using metrics such as accuracy, precision, recall, and confusion matrix analysis. The hypothesis is that a CNN-based approach can learn meaningful image features and provide reliable classification performance.",
    },
    {
        "title": "4. Methodology & Research Design",
        "why": "Reviewers ask this to see whether your process was systematic and justified. Methodology is where they check whether your work is credible and reproducible.",
        "questions": [
            "What steps did you follow?",
            "Why did you choose this methodology?",
            "Is the design experimental, analytical, survey-based, or system-based?",
            "How did you validate the work?",
            "Can someone reproduce your process?",
        ],
        "strong": "A strong answer describes the workflow step by step: data, preprocessing, model or design, implementation, evaluation, and validation.",
        "mistakes": [
            "Explaining only the tool, not the process.",
            "Skipping preprocessing or evaluation.",
            "Saying you used a method only because it is popular.",
            "Not explaining why the methodology fits the problem.",
        ],
        "example": "The methodology follows a system-development and experimental evaluation approach. First, MRI image data is collected and preprocessed, then a classification model is trained and tested using separated datasets. Finally, the model performance is evaluated using standard classification metrics to determine whether the approach is effective.",
    },
    {
        "title": "5. Data Collection & Sampling Strategy",
        "why": "Reviewers ask this to know whether your data is reliable, relevant, and enough to support your conclusions. Poor data weakens even a technically good project.",
        "questions": [
            "Where did the data come from?",
            "How many samples were used?",
            "How was the data selected?",
            "Was the dataset balanced or imbalanced?",
            "How did you split training and testing data?",
            "Are there any data quality issues?",
        ],
        "strong": "A strong answer names the data source, explains selection criteria, describes preprocessing, and mentions limitations such as class imbalance or dataset size.",
        "mistakes": [
            "Not knowing the dataset source.",
            "Ignoring class imbalance.",
            "Mixing training and testing data incorrectly.",
            "Claiming the dataset represents all real-world cases.",
        ],
        "example": "The dataset consists of MRI brain images collected from a publicly available medical imaging source. The images were divided into training and testing sets so the model could be evaluated on unseen data. I also considered preprocessing steps such as resizing and normalization to make the input consistent.",
    },
    {
        "title": "6. Tools, Technologies & Frameworks Used",
        "why": "Reviewers ask this to know whether you understand the technologies you used, rather than simply assembling code from different sources.",
        "questions": [
            "Which programming language and frameworks did you use?",
            "Why did you choose them?",
            "What role did each tool play?",
            "Could another tool have been used?",
            "What are the advantages and limitations of your chosen tools?",
        ],
        "strong": "A strong answer explains each tool's purpose and why it was suitable for the project.",
        "mistakes": [
            "Listing tools without explaining their role.",
            "Saying you chose a tool only because it was easy.",
            "Not understanding framework basics.",
            "Overemphasizing tools instead of the solution.",
        ],
        "example": "I used Python because it has strong support for machine learning and image processing. Libraries such as TensorFlow or Keras were used for model development, while NumPy, OpenCV, and Matplotlib helped with preprocessing and visualization. These tools were chosen because they are reliable, well-documented, and suitable for image classification tasks.",
    },
    {
        "title": "7. Results & Findings",
        "why": "Reviewers ask this to see what your project actually achieved. Results show whether your objectives were met.",
        "questions": [
            "What were the main results?",
            "Which metrics did you use?",
            "Did the system perform well?",
            "Were there any surprising findings?",
            "How do results compare with expectations?",
        ],
        "strong": "A strong answer reports results clearly using numbers, charts, or tables, and explains what those results mean.",
        "mistakes": [
            "Only saying the result was good.",
            "Reporting accuracy alone.",
            "Hiding weak results.",
            "Not explaining what the metrics indicate.",
        ],
        "example": "The model achieved promising classification performance on the test dataset, with accuracy supported by precision and recall values. The confusion matrix showed which classes were predicted well and where misclassification occurred. These findings suggest that the model learned useful image patterns, although further validation is needed before real clinical use.",
    },
    {
        "title": "8. Analysis & Interpretation",
        "why": "Reviewers ask this because results tell what happened, while analysis explains why it happened. They want to see your thinking, not just your output.",
        "questions": [
            "Why did the model or system perform this way?",
            "What patterns did you observe?",
            "What do the results imply?",
            "Are the results consistent with previous studies?",
            "What caused errors or limitations?",
        ],
        "strong": "A strong answer connects results to causes, theory, data quality, model behavior, or design decisions.",
        "mistakes": [
            "Repeating results without interpreting them.",
            "Making claims beyond the data.",
            "Ignoring errors.",
            "Not explaining misclassifications or unexpected outcomes.",
        ],
        "example": "The results indicate that the model was able to identify visual patterns associated with different tumor categories. Misclassifications may have occurred because some MRI images have similar textures or unclear boundaries. This suggests that better preprocessing, more data, or model tuning could improve reliability.",
    },
    {
        "title": "9. Limitations of the Study",
        "why": "Reviewers ask this to see academic honesty. Every project has limits, and acknowledging them shows maturity.",
        "questions": [
            "What are the weaknesses of your project?",
            "What assumptions did you make?",
            "What conditions affect the result?",
            "Can the system be used directly in the real world?",
            "What would you improve with more time?",
        ],
        "strong": "A strong answer admits limitations clearly but explains how they affect the work and how they could be addressed.",
        "mistakes": [
            "Saying there are no limitations.",
            "Listing only trivial limitations.",
            "Sounding apologetic or defensive.",
            "Mentioning limitations that destroy the whole project without context.",
        ],
        "example": "One limitation is that the model was trained on a limited dataset, so its performance may not generalize to all hospital environments. Another limitation is that the system depends on image quality and correct preprocessing. Therefore, the project should be seen as a supportive prototype rather than a final diagnostic tool.",
    },
    {
        "title": "10. Contributions & Novelty",
        "why": "Reviewers ask this to know what your work adds. Contribution does not always mean inventing something completely new; it can be a useful implementation, comparison, improvement, or integration.",
        "questions": [
            "What is your original contribution?",
            "How is your work different from existing systems?",
            "What did you design, implement, or evaluate?",
            "Is the novelty technical, practical, or analytical?",
            "Who benefits from your contribution?",
        ],
        "strong": "A strong answer states the contribution modestly and specifically.",
        "mistakes": [
            "Claiming the project is entirely new when similar work exists.",
            "Confusing effort with contribution.",
            "Saying your contribution is only coding the project.",
            "Not linking contribution to the identified gap.",
        ],
        "example": "The contribution of this project is the development of an end-to-end prototype for brain tumor image classification, including preprocessing, model training, evaluation, and result presentation. While the individual techniques are established, the value lies in integrating them into a focused, usable system. The project also demonstrates how deep learning can support medical image analysis in an educational or prototype setting.",
    },
    {
        "title": "11. Future Work & Scope",
        "why": "Reviewers ask this to know whether you understand how the project could grow beyond its current version.",
        "questions": [
            "What improvements can be made?",
            "Can the system scale?",
            "What features would you add later?",
            "How can accuracy or reliability improve?",
            "Can this be deployed in real-world settings?",
        ],
        "strong": "A strong answer gives realistic future improvements that directly relate to current limitations.",
        "mistakes": [
            "Suggesting unrelated future features.",
            "Saying only 'increase accuracy'.",
            "Making unrealistic claims like immediate hospital deployment.",
            "Not prioritizing future work.",
        ],
        "example": "Future work could include training the model on a larger and more diverse dataset to improve generalization. The system could also include explainability features, such as highlighting image regions that influenced the prediction. With proper medical validation, it could later be developed into a decision-support tool.",
    },
    {
        "title": "12. Ethical Considerations",
        "why": "Reviewers ask this to know whether you understand the responsibility involved, especially when working with health, user data, or automated decisions.",
        "questions": [
            "Was personal data involved?",
            "How was privacy protected?",
            "Could the system cause harm if misused?",
            "Are there bias or fairness concerns?",
            "Should humans remain involved in decisions?",
        ],
        "strong": "A strong answer recognizes privacy, consent, bias, transparency, and responsible use.",
        "mistakes": [
            "Saying ethics are not relevant.",
            "Ignoring privacy in medical data.",
            "Claiming the model can replace experts.",
            "Not mentioning misuse or bias.",
        ],
        "example": "Since this project deals with medical images, privacy and responsible use are important. The system should not be used as a replacement for doctors, but only as a supportive tool. Ethical deployment would require validated datasets, expert supervision, and safeguards against incorrect automated decisions.",
    },
    {
        "title": "13. Practical / Real-world Applications",
        "why": "Reviewers ask this to see whether your project has value beyond the classroom.",
        "questions": [
            "Where can this project be applied?",
            "Who are the users?",
            "What problem does it solve in practice?",
            "What changes are needed for real-world deployment?",
            "What are the risks in practical use?",
        ],
        "strong": "A strong answer gives realistic applications and avoids exaggerated claims.",
        "mistakes": [
            "Saying it can be used everywhere.",
            "Ignoring deployment challenges.",
            "Claiming clinical use without validation.",
            "Not identifying actual users.",
        ],
        "example": "This project could be useful in medical education, research demonstrations, or as an early-stage clinical decision-support prototype. In real-world healthcare, it could help radiologists by providing a preliminary classification, but only after strict validation. Practical use would require high-quality data, regulatory approval, and expert oversight.",
    },
    {
        "title": "14. Defense of Design Choices",
        "why": "Reviewers ask this to know whether your decisions were thoughtful. This is where they test whether you understand alternatives.",
        "questions": [
            "Why did you choose this approach?",
            "Why not use another model, tool, or method?",
            "What alternatives did you consider?",
            "What trade-offs did your approach involve?",
            "How do your choices affect performance and usability?",
        ],
        "strong": "A strong answer compares alternatives briefly and explains why your choice best fits your objectives, resources, and constraints.",
        "mistakes": [
            "Saying you chose it only because it was easy.",
            "Not knowing alternatives.",
            "Defending every choice emotionally.",
            "Ignoring trade-offs.",
        ],
        "example": "I chose a CNN-based approach because CNNs are well-suited for image classification tasks and can automatically learn spatial features from MRI images. Traditional machine learning methods usually require manual feature extraction, which may be less effective for complex medical images. The trade-off is that CNNs require more data and computational resources, so dataset quality and tuning are important.",
    },
]


algorithms = [
    {
        "title": "Image Preprocessing Algorithms",
        "points": [
            "Image resizing: converts all MRI images to the same size.",
            "Normalization: scales pixel values, usually from 0-255 to 0-1.",
            "Data augmentation: creates image variations using rotation, zoom, flip, shift, and similar transformations.",
            "Train-test split: divides data into training and testing sets so performance can be checked on unseen images.",
        ],
        "example": "Before training, I resized all images to a fixed dimension and normalized pixel values so the model could process them consistently. I also used data augmentation to reduce overfitting and help the model learn from different image variations.",
    },
    {
        "title": "Main Model: Convolutional Neural Network (CNN)",
        "points": [
            "Convolution layers extract visual features such as edges, textures, and tumor-related patterns.",
            "Pooling layers reduce image size while keeping important features.",
            "Flatten layer converts extracted features into one-dimensional form.",
            "Dense layers perform the final classification.",
            "Sigmoid output is commonly used for binary classification; softmax output is commonly used for multi-class classification.",
        ],
        "example": "The main model used in my project is a Convolutional Neural Network. CNNs are effective for image classification because they automatically extract important visual features from MRI images without requiring manual feature engineering.",
    },
    {
        "title": "Classification Type",
        "points": [
            "Binary classification is used when the system predicts tumor or no tumor.",
            "Multi-class classification is used when the system predicts classes such as glioma, meningioma, pituitary tumor, or no tumor.",
        ],
        "example": "My model performs image classification on MRI scans. Depending on the dataset classes, it can be presented as binary classification for tumor versus no tumor, or multi-class classification for tumor categories.",
    },
    {
        "title": "Training Algorithms and Techniques",
        "points": [
            "Backpropagation updates the model weights during training.",
            "Adam optimizer minimizes loss efficiently and is commonly used in deep learning image classification.",
            "Binary cross-entropy is used for two-class classification.",
            "Categorical cross-entropy is used for multi-class classification.",
        ],
        "example": "The model was trained using backpropagation with the Adam optimizer. The loss function was binary cross-entropy for two-class classification, or categorical cross-entropy if multiple tumor classes were used.",
    },
    {
        "title": "Evaluation Methods",
        "points": [
            "Accuracy measures overall correctness.",
            "Precision measures how many predicted tumor cases were actually correct.",
            "Recall measures how many actual tumor cases were detected.",
            "F1-score balances precision and recall.",
            "Confusion matrix shows correct and incorrect predictions for each class.",
        ],
        "example": "I evaluated the model using accuracy, precision, recall, F1-score, and confusion matrix. Accuracy shows overall correctness, while precision and recall help explain how well the model detects tumor cases specifically.",
    },
]


question_bank = [
    ("Basic", [
        "What is the main problem your project addresses?",
        "Why did you choose this topic?",
        "What are the main objectives of your project?",
        "Who are the intended users or beneficiaries?",
        "What tools and technologies did you use?",
        "What dataset or data source did you use?",
        "What are the main features of your system or study?",
    ]),
    ("Intermediate", [
        "What gap did you identify in existing work?",
        "How is your project different from previous studies or systems?",
        "Why did you choose this methodology?",
        "How did you preprocess or prepare your data?",
        "How did you evaluate your results?",
        "Why did you use these performance metrics?",
        "What were the major findings of your project?",
        "What limitations did you observe?",
    ]),
    ("Advanced", [
        "Why did you choose this model or approach instead of alternatives?",
        "How do you know your results are reliable?",
        "What would happen if your dataset were larger, smaller, or imbalanced?",
        "How would your system perform in a real-world environment?",
        "What ethical risks exist, and how would you reduce them?",
    ]),
]


def bullet_list(items, styles):
    return ListFlowable(
        [
            ListItem(Paragraph(item, styles["Body"]), leftIndent=12)
            for item in items
        ],
        bulletType="bullet",
        start="circle",
        leftIndent=18,
        bulletFontSize=7,
        bulletColor=colors.HexColor("#2f5f8f"),
    )


def add_label(story, label, text, styles):
    story.append(Paragraph(f"<b>{label}</b>", styles["Label"]))
    story.append(Paragraph(text, styles["Body"]))
    story.append(Spacer(1, 0.08 * inch))


def header_footer(canvas, doc):
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(colors.HexColor("#d8e2ee"))
    canvas.line(doc.leftMargin, height - 0.45 * inch, width - doc.rightMargin, height - 0.45 * inch)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#6b7280"))
    canvas.drawString(doc.leftMargin, 0.35 * inch, "Thesis / Project Review Preparation Guide")
    canvas.drawRightString(width - doc.rightMargin, 0.35 * inch, f"Page {doc.page}")
    canvas.restoreState()


def build_pdf():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="TitleCustom",
            parent=styles["Title"],
            alignment=TA_CENTER,
            fontName="Helvetica-Bold",
            fontSize=24,
            leading=30,
            textColor=colors.HexColor("#1f3552"),
            spaceAfter=16,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Subtitle",
            parent=styles["BodyText"],
            alignment=TA_CENTER,
            fontSize=11,
            leading=16,
            textColor=colors.HexColor("#4b5563"),
            spaceAfter=22,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionTitle",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=19,
            textColor=colors.HexColor("#204a73"),
            spaceBefore=12,
            spaceAfter=7,
            keepWithNext=True,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Label",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=13,
            textColor=colors.HexColor("#1f2937"),
            spaceBefore=5,
            spaceAfter=2,
            keepWithNext=True,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Body",
            parent=styles["BodyText"],
            alignment=TA_LEFT,
            fontName="Helvetica",
            fontSize=9.5,
            leading=13.5,
            textColor=colors.HexColor("#263238"),
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Example",
            parent=styles["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=9.2,
            leading=13,
            textColor=colors.HexColor("#374151"),
            leftIndent=10,
            rightIndent=8,
            borderColor=colors.HexColor("#c8d8ea"),
            borderWidth=0.6,
            borderPadding=7,
            backColor=colors.HexColor("#f5f8fc"),
            spaceBefore=2,
            spaceAfter=8,
        )
    )

    doc = SimpleDocTemplate(
        OUTPUT_PATH,
        pagesize=A4,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.65 * inch,
        title="Thesis Review Preparation Guide - BrainTumorAI",
        author="Codex",
    )

    story = []
    story.append(Paragraph("Thesis / Project Review Preparation Guide", styles["TitleCustom"]))
    story.append(
        Paragraph(
            "Prepared for a Brain Tumor AI project review. Use this as a calm rehearsal sheet: clear answers, honest limitations, and confident reasoning matter more than memorized perfection.",
            styles["Subtitle"],
        )
    )

    for section in sections:
        story.append(Paragraph(section["title"], styles["SectionTitle"]))
        add_label(story, "Why reviewers ask about this", section["why"], styles)
        story.append(Paragraph("<b>Key sub-questions reviewers typically raise</b>", styles["Label"]))
        story.append(bullet_list(section["questions"], styles))
        story.append(Spacer(1, 0.04 * inch))
        add_label(story, "What a strong, well-prepared answer looks like", section["strong"], styles)
        story.append(Paragraph("<b>Common mistakes students make</b>", styles["Label"]))
        story.append(bullet_list(section["mistakes"], styles))
        story.append(Spacer(1, 0.04 * inch))
        story.append(Paragraph("<b>Model 2-3 sentence answer</b>", styles["Label"]))
        story.append(Paragraph(section["example"], styles["Example"]))

    story.append(PageBreak())
    story.append(Paragraph("Algorithms and Models Used", styles["SectionTitle"]))
    story.append(
        Paragraph(
            "For a Brain Tumor AI or MRI image classification project, reviewers usually expect you to explain both the preprocessing algorithms and the deep learning model. The answer should make clear what each part does and why it was chosen.",
            styles["Body"],
        )
    )
    for item in algorithms:
        story.append(Paragraph(item["title"], styles["SectionTitle"]))
        story.append(bullet_list(item["points"], styles))
        story.append(Spacer(1, 0.04 * inch))
        story.append(Paragraph("<b>Example answer</b>", styles["Label"]))
        story.append(Paragraph(item["example"], styles["Example"]))

    story.append(Paragraph("Strong Short Answer for Viva", styles["SectionTitle"]))
    story.append(
        Paragraph(
            "My project uses a CNN-based deep learning model for brain tumor classification from MRI images. The images are first preprocessed using resizing, normalization, and augmentation, then the CNN extracts visual features through convolution and pooling layers. The model is trained using backpropagation with the Adam optimizer and evaluated using accuracy, precision, recall, F1-score, and confusion matrix.",
            styles["Example"],
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("Reviewer Question Bank", styles["SectionTitle"]))
    story.append(
        Paragraph(
            "These questions are sorted from basic to advanced. Practice answering them aloud in simple language, then add your project-specific details.",
            styles["Body"],
        )
    )
    number = 1
    for level, questions in question_bank:
        story.append(Paragraph(level, styles["SectionTitle"]))
        numbered = []
        for question in questions:
            numbered.append(f"{number}. {question}")
            number += 1
        story.append(bullet_list(numbered, styles))

    story.append(Spacer(1, 0.16 * inch))
    story.append(
        Paragraph(
            "Final reminder: a good review answer does not need to sound perfect. It should sound clear, honest, and owned by you. If you can explain what you did, why you did it, what you found, and what you would improve, you are already in strong shape.",
            styles["Example"],
        )
    )

    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)


if __name__ == "__main__":
    build_pdf()
    print(OUTPUT_PATH)
